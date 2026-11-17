#!/usr/bin/env python3
"""Analyze complete study arms without fitting, inference or training-state writes."""

import argparse
import datetime
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from drone_detector.adaptation import cohort_signature
from drone_detector.common import digest, file_hash, read_json, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from coordinate_beats_arms import ARMS  # noqa: E402
from report_beats_adaptation import batches, process_active, verified_result  # noqa: E402

ANALYSIS_CODE = [
    "scripts/compare_completed_beats_arm.py",
    "src/drone_detector/paired_comparison.py",
    "src/drone_detector/beats_training.py",
    "scripts/train_beats_adaptation.py",
]


def atomic_json(path, value):
    temporary = path.with_suffix(".partial")
    write_json(temporary, value)
    temporary.replace(path)


def effective_status(state, active):
    status = state["status"]
    return "interrupted" if status in {"running", "pausing"} and not active else status


def cached_pair(output, private, expected, frozen, adapted):
    """Reuse only a complete analysis tied to the same inputs and fitted models."""
    path = output / "paired-comparison.json"
    if not path.exists():
        return False
    summary = read_json(path)
    if (
        any(summary.get(key) != value for key, value in expected.items())
        or summary["frozen_evaluation"] != frozen
        or summary["adapted_evaluation"] != adapted
        or file_hash(private / "paired-predictions.jsonl")
        != summary["private_paired_ledger_sha256"]
        or not (output / "README.md").is_file()
    ):
        raise ValueError("Existing paired analysis has changed; preserve it and investigate")
    return True


class Observer:
    def __init__(self, args):
        self.args = args
        self.plan_sha = file_hash(args.work / "plan.json")
        self.manifest_sha = file_hash(args.work / "manifest.jsonl")
        self.plan = read_json(args.work / "plan.json")
        self.rows = read_jsonl(args.work / "manifest.jsonl")
        if cohort_signature(self.rows) != self.plan["recipe"]["cohort_signature"]:
            raise ValueError("Observer cohort differs from the declared study")
        self.code = {name: file_hash(ROOT / name) for name in ANALYSIS_CODE}
        self.seen = {}
        self.summaries = {}
        self.missing_observations = 0
        self.last_status = None

    def command(self, name, *arguments):
        subprocess.run(
            [str(self.args.python), "-u", str(ROOT / "scripts" / name), *map(str, arguments)],
            cwd=ROOT,
            check=True,
            env=dict(
                os.environ,
                PYTHONPATH=str(ROOT / "src"),
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
            ),
        )

    def observe_arm(self, fold, mix):
        folder = self.args.work / fold / mix / "fine-tuned"
        validation = folder / "validation.json"
        if not validation.exists() or time.time() - validation.stat().st_mtime < 2:
            return False
        # Completed artifacts are immutable. Hash small receipts each poll; read
        # the large selected encoder once on first verification of each arm.
        identity = {
            name: file_hash(folder / name)
            for name in [
                "validation.json",
                "training-complete.json",
                "model.json",
                "head.npz",
                "best.pt.json",
            ]
        }
        stat = (folder / "best.pt").stat()
        identity["selected_encoder_stat"] = [stat.st_size, stat.st_mtime_ns]
        key = fold + "/" + mix
        if key in self.seen:
            if identity != self.seen[key]:
                raise ValueError("A previously analyzed completed arm changed: " + key)
            output = self.args.report / "paired-analysis" / fold / mix
            private = self.args.private / fold / mix
            receipt = self.summaries[key]
            if (
                file_hash(output / "paired-comparison.json") != receipt["paired_comparison_sha256"]
                or file_hash(private / "paired-predictions.jsonl")
                != receipt["private_ledger_sha256"]
            ):
                raise ValueError("A previously verified paired analysis changed: " + key)
            return False
        ids = self.plan["folds"][fold][mix + "_positive_train"]
        recipe = self.plan["recipe"]["fine_tuning"]
        steps = batches(self.rows, ids, recipe["batch_size"]) * recipe["epochs"]
        results = {
            family: verified_result(
                self.args.work,
                fold,
                mix,
                family,
                self.plan_sha,
                steps,
                len(ids),
                digest(ids),
                recipe,
            )
            for family in ["frozen", "fine-tuned"]
        }
        if any(value is None for value in results.values()):
            raise ValueError("A complete matched control and adapted arm are required")
        base = self.args.work / "base-embeddings.npz"
        adapted = folder / "embeddings.npz"
        expected = dict(
            held_source=fold,
            positive_mix=mix,
            corpus_rows=len(self.rows),
            compared_validation_rows=len(self.plan["folds"][fold]["validation"]),
            optimizer_steps=steps,
            plan_sha256=self.plan_sha,
            manifest_sha256=self.manifest_sha,
            code_sha256=self.code,
            base_embeddings_sha256=file_hash(base),
            adapted_embeddings_sha256=file_hash(adapted),
        )
        output = self.args.report / "paired-analysis" / fold / mix
        private = self.args.private / fold / mix
        if not cached_pair(output, private, expected, results["frozen"], results["fine-tuned"]):
            self.command(
                "compare_completed_beats_arm.py",
                "--work",
                self.args.work,
                "--fold",
                fold,
                "--positive-mix",
                mix,
                "--output",
                output,
                "--private-output",
                private,
            )
            if not cached_pair(output, private, expected, results["frozen"], results["fine-tuned"]):
                raise ValueError("Completed analysis did not produce a verified result")
        self.seen[key] = identity
        self.summaries[key] = dict(
            paired_comparison_sha256=file_hash(output / "paired-comparison.json"),
            private_ledger_sha256=file_hash(private / "paired-predictions.jsonl"),
            compared_validation_rows=expected["compared_validation_rows"],
        )
        print("Verified complete paired analysis", key, flush=True)
        return True

    def poll(self, refresh=False):
        if (
            file_hash(self.args.work / "plan.json") != self.plan_sha
            or file_hash(self.args.work / "manifest.jsonl") != self.manifest_sha
            or any(file_hash(ROOT / name) != value for name, value in self.code.items())
        ):
            raise ValueError("Observer inputs or analysis implementation changed")
        changed = False
        for fold, mix in ARMS:
            changed = self.observe_arm(fold, mix) or changed
        state = read_json(self.args.work / "pipeline-state.json")
        active = process_active(state.get("child_pid"), self.args.work)
        status = effective_status(state, active)
        if status == "complete" and len(self.seen) != len(ARMS):
            raise ValueError("Study claims completion without all four verified paired analyses")
        self.missing_observations = self.missing_observations + 1 if status == "interrupted" else 0
        # A supervisor can briefly have no child between stages. Confirm a
        # missing child on a second poll before stopping this observer.
        terminal = status in {"complete", "paused", "failed"} or self.missing_observations >= 2
        if changed or refresh or terminal or status != self.last_status:
            self.command(
                "report_beats_adaptation.py", "--work", self.args.work, "--output", self.args.report
            )
        self.last_status = status
        receipt = dict(
            captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            status="stopped" if terminal else "watching",
            study_status=status,
            active_training_child_verified=active,
            completed_analyses=self.summaries,
            completed_arms=len(self.seen),
            planned_arms=len(ARMS),
            plan_sha256=self.plan_sha,
            manifest_sha256=self.manifest_sha,
            accumulated_hours=state["accumulated_seconds"] / 3600,
            max_wall_hours=state.get("max_wall_hours", 44),
            observer_sha256=file_hash(__file__),
            analysis_code_sha256=self.code,
            scope="Read-only study observer; complete-cohort analysis at locked cutoffs. No fit, encoder inference, training restart, budget extension or remote publication.",
        )
        atomic_json(self.args.state, receipt)
        return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["work", "report", "private", "state", "python"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 5 <= args.poll_seconds <= 60:
        parser.error("Use a poll interval between 5 and 60 seconds")
    for name in ["work", "report", "private", "state", "python"]:
        setattr(args, name, getattr(args, name).resolve())
    # Outputs must never point inside the study that this process observes.
    for output in [args.report, args.private, args.state]:
        if output == args.work or args.work in output.parents:
            parser.error("Observer outputs must be outside the training study")
    if (
        args.private == args.report
        or args.report in args.private.parents
        or args.private in args.report.parents
    ):
        parser.error("Private ledgers and aggregate reports require separate directories")
    lock_root = Path(tempfile.gettempdir()) / "drone-completion-observer-locks"
    lock_root.mkdir(exist_ok=True)
    lock_path = lock_root / hashlib.sha256(str(args.work).encode()).hexdigest()
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.nice(10)
        try:
            observer = Observer(args)
            first = True
            while True:
                terminal = observer.poll(refresh=first)
                if terminal or args.once:
                    return
                first = False
                time.sleep(args.poll_seconds)
        except Exception as error:
            atomic_json(
                args.state,
                dict(
                    status="failed",
                    error=str(error),
                    scope="Observer failure; training was not modified",
                ),
            )
            raise


if __name__ == "__main__":
    main()
