#!/usr/bin/env python3
"""Reuse verified completed arms; delegate unfinished work to the pinned driver."""

import argparse
from pathlib import Path
import sys

from drone_detector.adaptation import cohort_signature
from drone_detector.common import digest, file_hash, read_json, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from report_beats_adaptation import batches, verified_result  # noqa: E402

DRIVER_SHA256 = "201d6f2e6be1494c203c5586cd2dec64fa19cfc350fbec72f774e1e5aa835f92"
TRAINER_SHA256 = "c8fc62ae8085b52073a67cb662f22a908042f020cfac038a9d5059b50750c419"
ARMS = [
    ("audrok-dresden", "current"),
    ("audrok-dresden", "broader"),
    ("wsu-air3", "current"),
    ("wsu-air3", "broader"),
]


def coordinate(load_completed, execute):
    results = {"audrok-dresden": {}, "wsu-air3": {}}
    for fold, mix in ARMS:
        result = load_completed(fold, mix)
        if result is None:
            result = execute(fold, mix)
        else:
            print("Reusing verified completed adaptation arm", fold, mix, flush=True)
        if result.get("status") == "paused":
            return None
        results[fold][mix] = result
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--keep-completed-optimizer", action="store_true")
    args = parser.parse_args()
    if (
        file_hash(ROOT / "scripts/train_beats_adaptation.py") != DRIVER_SHA256
        or file_hash(ROOT / "src/drone_detector/beats_training.py") != TRAINER_SHA256
    ):
        raise ValueError("Recovery requires the unchanged numerical driver and trainer")
    # Import the original module, keeping its __file__, source identities,
    # model construction, optimizer, RNG restoration and arithmetic unchanged.
    import train_beats_adaptation as driver

    driver.configure(args.device, threads=args.threads)
    rows = read_jsonl(args.work / "manifest.jsonl")
    saved = read_json(args.work / "plan.json")
    if cohort_signature(rows) != saved["recipe"]["cohort_signature"]:
        raise ValueError("Study cohort changed")
    plan_sha = file_hash(args.work / "plan.json")
    recipe = saved["recipe"]["fine_tuning"]

    def completed(fold, mix):
        ids = saved["folds"][fold][mix + "_positive_train"]
        steps = batches(rows, ids, recipe["batch_size"]) * recipe["epochs"]
        return verified_result(
            args.work, fold, mix, "fine-tuned", plan_sha, steps, len(ids), digest(ids), recipe
        )

    results = coordinate(completed, lambda fold, mix: driver.run_arm(args, fold, mix))
    if results is None:
        raise SystemExit(75)
    write_json(args.work / "fine-tuned-controls.json", results)


if __name__ == "__main__":
    main()
