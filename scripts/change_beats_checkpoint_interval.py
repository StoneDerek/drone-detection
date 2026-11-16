#!/usr/bin/env python3
"""Change only recovery cadence before the first completed adaptation arm.

The trainer and training driver stay byte-identical to the original run. A
paused legacy launcher is migrated explicitly; a shorter interval does not
change data order, model/optimizer tensors, RNG, epochs or the remaining budget.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import torch

from drone_detector.beats_training import read_state, save_state
from drone_detector.common import file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
LEGACY_LAUNCHER_SHA = "336445354eb89737227e6aa0d81ebb8f2029e6f4d3665aa73e8e032914af07ae"


def training_fingerprint(state):
    """Hash every non-identity field, including optimizer tensors and all RNG."""
    h = hashlib.sha256()

    def visit(value):
        h.update((type(value).__name__ + ":").encode())
        if isinstance(value, torch.Tensor):
            h.update(str((value.dtype, tuple(value.shape))).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            h.update(str(len(value)).encode())
            for item in value:
                visit(item)
        elif isinstance(value, float):
            h.update(struct.pack("!d", value))
        else:
            h.update(json.dumps(value, ensure_ascii=True).encode())

    visit({key: value for key, value in state.items() if key != "identity"})
    return h.hexdigest()


def change_arm(folder, interval):
    state_path = folder / "latest.pt"
    state = read_state(state_path)
    original = dict(state["identity"])
    for key, source in [
        ("training_implementation_sha256", ROOT / "src/drone_detector/beats_training.py"),
        ("driver_sha256", ROOT / "scripts/train_beats_adaptation.py"),
    ]:
        if original[key] != file_hash(source):
            raise ValueError("Only an unchanged numerical trainer/driver may change cadence")
    marker = folder / "checkpoint-cadence-change.json"
    if marker.exists():
        receipt = read_json(marker)
        if receipt["new_identity"]["checkpoint_every_steps"] != interval:
            raise ValueError("An existing cadence migration targets another interval")
        if state["identity"] not in [receipt["old_identity"], receipt["new_identity"]]:
            raise ValueError("Checkpoint identity differs from the recorded migration")
    else:
        target = dict(original, checkpoint_every_steps=interval)
        receipt = dict(
            old_identity=original,
            new_identity=target,
            original_checkpoint_sha256=file_hash(state_path),
            unchanged_training_fingerprint=training_fingerprint(state),
            steps=state["steps"],
            epoch=state["epoch"],
            next_batch=state["position"],
        )
        write_json(marker, receipt)
    if training_fingerprint(state) != receipt["unchanged_training_fingerprint"]:
        raise ValueError("Model, optimizer, RNG or training counters changed")
    if state["identity"] != receipt["new_identity"]:
        state["identity"] = receipt["new_identity"]
        save_state(state_path, state)
    del state
    restored = read_state(state_path)
    if (
        restored["identity"] != receipt["new_identity"]
        or training_fingerprint(restored) != receipt["unchanged_training_fingerprint"]
    ):
        raise ValueError("Cadence migration changed training state")
    temporary = folder / "identity.partial"
    write_json(temporary, receipt["new_identity"])
    temporary.replace(folder / "identity.json")
    write_json(marker, dict(receipt, migrated_checkpoint_sha256=file_hash(state_path), verified=True))
    return dict(
        held_source=folder.parent.parent.name,
        positive_mix=folder.parent.name,
        steps=receipt["steps"],
        training_state_unchanged=True,
        unchanged_training_fingerprint=receipt["unchanged_training_fingerprint"],
        original_checkpoint_sha256=receipt["original_checkpoint_sha256"],
        migrated_checkpoint_sha256=file_hash(state_path),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--checkpoint-every", type=int, required=True)
    args = p.parse_args()
    work = args.work.resolve()
    if args.checkpoint_every < 1:
        p.error("Positive checkpoint interval required")
    state_path = work / "pipeline-state.json"
    state = read_json(state_path)
    if state["status"] != "paused" or not (work / "pause-requested.json").exists():
        raise ValueError("Cooperatively pause the study before changing recovery cadence")
    if list(work.rglob("training-complete.json")):
        raise ValueError("This migration is restricted to before the first completed arm")
    # Pause is published only after the training child exits. Refuse another
    # writer even when its PID comes from a different launch receipt.
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            arguments = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError):
            continue
        if str(work).encode() in arguments and any(
            value.endswith((b"/scripts/train_beats_adaptation.py", b"/scripts/launch_beats_adaptation.py"))
            for value in arguments
        ):
            raise ValueError("A study writer remains active")
    old = state["identity"]
    target = dict(
        old,
        launcher_sha256=file_hash(ROOT / "scripts/launch_beats_adaptation.py"),
        checkpoint_every_steps=args.checkpoint_every,
    )
    if old["launcher_sha256"] not in {LEGACY_LAUNCHER_SHA, target["launcher_sha256"]}:
        raise ValueError("This tool accepts only the recorded legacy or current launcher")
    changes = []
    for latest in sorted(work.glob("*/*/fine-tuned/latest.pt")):
        changes.append(change_arm(latest.parent, args.checkpoint_every))
    if not changes:
        raise ValueError("A saved adaptation optimizer checkpoint is required")
    audit = dict(
        old_pipeline_identity=old,
        new_pipeline_identity=target,
        unchanged_accumulated_seconds=state["accumulated_seconds"],
        checkpoint_every_steps=args.checkpoint_every,
        changed_arms=changes,
        numerical_trainer_sha256=file_hash(ROOT / "src/drone_detector/beats_training.py"),
        numerical_driver_sha256=file_hash(ROOT / "scripts/train_beats_adaptation.py"),
    )
    write_json(work / "checkpoint-cadence-change.json", audit)
    temporary = state_path.with_suffix(".partial")
    write_json(temporary, dict(state, identity=target))
    temporary.replace(state_path)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
