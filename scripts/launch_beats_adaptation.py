#!/usr/bin/env python3
"""Run all full-cohort stages with an explicit, cumulative wall-time budget."""

import argparse
import fcntl
import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path

from drone_detector.common import file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


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
    p.add_argument("--max-wall-hours", type=float, default=44)
    args = p.parse_args()
    if args.max_wall_hours <= 0 or args.threads < 1:
        p.error("Positive time budget and thread count required")
    args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    identity = dict(
        plan_sha256=file_hash(args.work / "plan.json"),
        device=args.device,
        threads=args.threads,
        launcher_sha256=file_hash(__file__),
    )
    state_path = args.work / "pipeline-state.json"
    state = (
        read_json(state_path)
        if state_path.exists()
        else dict(identity=identity, accumulated_seconds=0)
    )
    if state["identity"] != identity:
        raise ValueError("Existing pipeline uses another protocol or execution runtime")
    if state.get("status") == "complete":
        print("The full study has already completed", flush=True)
        return
    old_pid = state.get("child_pid")
    if state.get("status") in {"running", "pausing"} and old_pid:
        try:
            command = Path(f"/proc/{old_pid}/cmdline").read_bytes()
            if str(args.work).encode() in command:
                raise RuntimeError(
                    "The previous study child is still active; do not start another writer"
                )
        except FileNotFoundError:
            pass
    lock_root = Path(tempfile.gettempdir()) / "drone-adaptation-pipeline-locks"
    lock_root.mkdir(exist_ok=True)
    lock_path = lock_root / hashlib.sha256(str(args.work).encode()).hexdigest()
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another pipeline writer is active on this runtime") from error
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
            prefix + [str(ROOT / "scripts/train_beats_adaptation.py"), "all", *native],
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
