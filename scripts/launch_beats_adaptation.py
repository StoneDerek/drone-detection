#!/usr/bin/env python3
"""Run all full-cohort stages with an explicit, cumulative wall-time budget."""

import argparse
import fcntl
import hashlib
import copy
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

from drone_detector.common import file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
LEGACY_LAUNCHER_SHA256 = "6ed36e9ad538fe80cb5f191790de33dd58be4e907f877a9357b3d12fc6a8fce1"


def child_active(pid, work):
    if not isinstance(pid, int):
        return False
    try:
        proc = Path(f"/proc/{pid}")
        arguments = (proc / "cmdline").read_bytes().split(b"\0")
        state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, PermissionError):
        return False
    return (
        state != "Z"
        and str(work.resolve()).encode() in arguments
        and any(
            value.endswith(b"/scripts/train_beats_adaptation.py")
            or value.endswith(b"/scripts/coordinate_beats_arms.py")
            or value.endswith(b"/scripts/run_beats_adaptation.py")
            for value in arguments
        )
    )


def migrated_supervision(state, identity, max_hours, active=False, pause_exists=False):
    """Only replace supervision identity; never edit a neural checkpoint."""
    old = state["identity"]
    if (
        active
        or pause_exists
        or state.get("status") not in {"running", "failed", "interrupted"}
        or old.get("launcher_sha256") != LEGACY_LAUNCHER_SHA256
    ):
        raise ValueError(
            "Existing study is active, paused, complete or not the supported legacy launcher"
        )
    old_protocol = {key: value for key, value in old.items() if key != "launcher_sha256"}
    new_protocol = {
        key: value
        for key, value in identity.items()
        if key not in {"launcher_sha256", "arm_coordinator_sha256"}
    }
    elapsed = state["accumulated_seconds"]
    if (
        old_protocol != new_protocol
        or state.get("max_wall_hours") != max_hours
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ValueError("Migration cannot change protocol, runtime or cumulative budget")
    result = copy.deepcopy(state)
    result.update(identity=identity, status="interrupted")
    return result


def publish(path, value):
    temporary = path.with_suffix(".partial")
    write_json(temporary, value)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--max-wall-hours", type=float, default=44)
    p.add_argument("--migrate-legacy-supervision", action="store_true")
    args = p.parse_args()
    if args.max_wall_hours <= 0 or args.threads < 1 or args.checkpoint_every < 1:
        p.error("Positive time budget, thread count and checkpoint interval required")
    args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    identity = dict(
        plan_sha256=file_hash(args.work / "plan.json"),
        device=args.device,
        threads=args.threads,
        checkpoint_every_steps=args.checkpoint_every,
        launcher_sha256=file_hash(__file__),
        arm_coordinator_sha256=file_hash(ROOT / "scripts/coordinate_beats_arms.py"),
    )
    state_path = args.work / "pipeline-state.json"
    state = (
        read_json(state_path)
        if state_path.exists()
        else dict(identity=identity, accumulated_seconds=0)
    )
    if state["identity"] != identity and not args.migrate_legacy_supervision:
        raise ValueError("Existing pipeline uses another protocol or execution runtime")
    if state.get("status") == "complete":
        print("The full study has already completed", flush=True)
        return
    old_pid = state.get("child_pid")
    if child_active(old_pid, args.work):
        raise RuntimeError("The previous study child is still active; do not start another writer")
    lock_root = Path(tempfile.gettempdir()) / "drone-adaptation-pipeline-locks"
    lock_root.mkdir(exist_ok=True)
    lock_path = lock_root / hashlib.sha256(str(args.work).encode()).hexdigest()
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another pipeline writer is active on this runtime") from error
        if state["identity"] != identity:
            migrated = migrated_supervision(
                state,
                identity,
                args.max_wall_hours,
                active=child_active(old_pid, args.work),
                pause_exists=(args.work / "pause-requested.json").exists(),
            )
            receipt = args.work / (
                "supervision-migration-" + identity["launcher_sha256"][:12] + ".json"
            )
            if receipt.exists():
                raise ValueError(
                    "Preserve the existing migration receipt before attempting another migration"
                )
            write_json(
                receipt,
                dict(
                    before=state,
                    after=migrated,
                    scope="Supervision metadata only. Neural driver, trainer and checkpoint files are not modified.",
                ),
            )
            publish(state_path, migrated)
            state = migrated
        run(args, identity, state, state_path)


def run(args, identity, state, state_path):
    elapsed_before = state["accumulated_seconds"]
    started = time.monotonic()
    pause = args.work / "pause-requested.json"
    env = dict(
        os.environ, PYTHONPATH=str(ROOT / "src"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"
    )
    prefix = [str(args.python), "-u"]
    native = [
        "--work",
        str(args.work),
        "--assets",
        str(args.assets),
        "--device",
        args.device,
        "--threads",
        str(args.threads),
        "--checkpoint-every",
        str(args.checkpoint_every),
    ]
    stages = [
        (
            "native-base-embeddings",
            prefix + [str(ROOT / "scripts/train_beats_adaptation.py"), "base", *native],
        ),
        (
            "paired-frozen-controls",
            prefix
            + [
                str(ROOT / "scripts/run_beats_adaptation.py"),
                "frozen",
                "--work",
                str(args.work),
                "--report",
                str(args.work / "report"),
            ],
        ),
        (
            "four-full-fine-tuning-arms",
            prefix + [str(ROOT / "scripts/coordinate_beats_arms.py"), *native],
        ),
    ]
    for name, command in stages:
        elapsed = elapsed_before + time.monotonic() - started
        if elapsed >= args.max_wall_hours * 3600 and not pause.exists():
            write_json(pause, dict(reason="cumulative wall-time budget", elapsed_seconds=elapsed))
        if pause.exists():
            publish(
                state_path,
                dict(identity=identity, status="paused", stage=name, accumulated_seconds=elapsed),
            )
            print("Study paused before next stage", name, flush=True)
            return
        logfile = args.work / (name + ".log")
        print("Starting complete-cohort stage", name, flush=True)
        with logfile.open("a") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            while process.poll() is None:
                elapsed = elapsed_before + time.monotonic() - started
                if elapsed >= args.max_wall_hours * 3600 and not pause.exists():
                    write_json(
                        pause, dict(reason="cumulative wall-time budget", elapsed_seconds=elapsed)
                    )
                publish(
                    state_path,
                    dict(
                        identity=identity,
                        status="pausing" if pause.exists() else "running",
                        stage=name,
                        child_pid=process.pid,
                        accumulated_seconds=elapsed,
                        max_wall_hours=args.max_wall_hours,
                    ),
                )
                time.sleep(5)
        elapsed = elapsed_before + time.monotonic() - started
        if process.returncode != 0:
            status = "paused" if pause.exists() else "failed"
            publish(
                state_path,
                dict(
                    identity=identity,
                    status=status,
                    stage=name,
                    accumulated_seconds=elapsed,
                    returncode=process.returncode,
                ),
            )
            if status == "failed":
                raise RuntimeError(
                    "Full study stage failed; preserve checkpoints and inspect " + str(logfile)
                )
            return
    publish(
        state_path,
        dict(
            identity=identity,
            status="complete",
            accumulated_seconds=elapsed_before + time.monotonic() - started,
        ),
    )
    print(
        "All four full fine-tuning arms and their held-source development scores completed",
        flush=True,
    )


if __name__ == "__main__":
    main()
