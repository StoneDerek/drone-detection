#!/usr/bin/env python3
"""Resume corpus preparation, fit frozen heads, and consume the final cohort once.

Acquisition can run concurrently in a different process. Each stage leaves
inspectable receipts; unavailable or corrupt data causes failure, not a reduced
five-file run. Encoder runtimes and assets are supplied explicitly.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from drone_detector.common import file_hash, read_json, write_json
from drone_detector.data import load_manifest

ROOT = Path(__file__).resolve().parents[1]


def checked(command, log):
    print(json.dumps({"stage": "starting", "command": [str(v) for v in command]}), flush=True)
    with log.open("a") as output:
        subprocess.run(
            [str(v) for v in command], cwd=ROOT, check=True, stdout=output, stderr=subprocess.STDOUT
        )


def covered(prepared, features, splits):
    rows = [
        r
        for r in load_manifest(prepared / "manifest.jsonl", require_split=True)
        if r["split"] in splits
    ]
    missing = {}
    for family in ["spectral", "clap", "beats"]:
        path = features / family / "index.json"
        index = read_json(path) if path.exists() else {}
        missing[family] = sum(r["id"] not in index for r in rows)
    return missing


def feature_command(args, family, splits):
    runtime = {"spectral": args.python, "beats": args.beats_python, "clap": args.clap_python}[
        family
    ]
    if args.feature_shards > 1:
        command = [
            args.python,
            ROOT / "scripts/shard_corpus_features.py",
            "--prepared",
            args.study / "prepared",
            "--output",
            args.study / "features" / family,
            "--workspace",
            args.study / "feature-shards",
            "--python",
            runtime,
            "--family",
            family,
            "--splits",
            *splits,
            "--shards",
            str(args.feature_shards),
            "--device",
            args.device,
        ]
        if family != "spectral":
            command += ["--assets", getattr(args, family + "_assets")]
        return command
    command = [
        runtime,
        ROOT / "scripts/extract_corpus_study_features.py",
        "--prepared",
        args.study / "prepared",
        "--output",
        args.study / "features" / family,
        "--family",
        family,
        "--splits",
        *splits,
        "--threads",
        "1",
        "--device",
        args.device,
    ]
    if family != "spectral":
        command += ["--assets", getattr(args, family + "_assets")]
    return command


def run(args):
    args.study = args.study.resolve()
    args.study.mkdir(parents=True, exist_ok=True)
    prepared = args.study / "prepared"
    features = args.study / "features"
    models = args.study / "models"
    if not (prepared / "preparation-complete.json").exists():
        if args.attach_preparation:
            while (
                not (prepared / "inventory.json").exists()
                or read_json(prepared / "inventory.json")["pending"]
            ):
                print("Waiting for complete source preparation", flush=True)
                time.sleep(30)
        else:
            checked(
                [
                    args.python,
                    ROOT / "scripts/prepare_corpus_study.py",
                    "--archives",
                    args.study / "archives",
                    "--output",
                    prepared,
                    "--urban-csv",
                    args.urban_csv,
                    "--wait",
                ],
                args.study / "preparation.log",
            )
        if read_json(prepared / "inventory.json")["pending"]:
            raise ValueError("Incomplete corpus inventory; a smaller fallback is not permitted")
        checked(
            [
                args.python,
                ROOT / "scripts/finalize_corpus_study.py",
                "--prepared",
                prepared,
                "--archives",
                args.study / "archives",
            ],
            args.study / "finalization.log",
        )
    if not (models / "frozen-models.json").exists():
        if args.attach_features:
            while True:
                missing = covered(prepared, features, {"train", "development"})
                print(json.dumps({"missing_train_development_features": missing}), flush=True)
                if not any(missing.values()):
                    break
                time.sleep(30)
            for pid in args.attached_feature_pids:
                while Path(f"/proc/{pid}").exists():
                    print(f"Waiting for attached feature writer {pid} to exit", flush=True)
                    time.sleep(5)
        else:
            for family in ["spectral", "clap", "beats"]:
                checked(
                    feature_command(args, family, ["train", "development"]),
                    args.study / (family + "-features.log"),
                )
        checked(
            [
                args.python,
                ROOT / "scripts/run_corpus_study.py",
                "freeze",
                "--prepared",
                prepared,
                "--features",
                features,
                "--output",
                models,
            ],
            args.study / "fitting.log",
        )
    if not (models / "comparison.json").exists():
        if (models / "final-use.json").exists():
            raise ValueError("A final use already began without completing; inspect saved state")
        workers = []
        handles = []
        try:
            for family in ["spectral", "clap", "beats"]:
                log = (args.study / (family + "-final-features.log")).open("a")
                handles.append(log)
                worker = subprocess.Popen(
                    [str(v) for v in feature_command(args, family, ["final"])],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                workers.append((family, worker))
            while True:
                failed = [
                    (family, worker.returncode)
                    for family, worker in workers
                    if worker.poll() not in {None, 0}
                ]
                if failed:
                    raise RuntimeError(f"Feature extraction failed: {failed}")
                missing = covered(prepared, features, {"final"})
                print(json.dumps({"missing_final_features": missing}), flush=True)
                if all(worker.poll() == 0 for _, worker in workers):
                    if any(missing.values()):
                        raise ValueError("Completed workers left missing holdout features")
                    break
                time.sleep(30)
        finally:
            for _, worker in workers:
                if worker.poll() is None:
                    worker.terminate()
            for handle in handles:
                handle.close()
        checked(
            [
                args.python,
                ROOT / "scripts/run_corpus_study.py",
                "evaluate",
                "--prepared",
                prepared,
                "--features",
                features,
                "--output",
                models,
            ],
            args.study / "evaluation.log",
        )
    write_json(
        args.study / "milestone-complete.json",
        {
            "manifest_sha256": file_hash(prepared / "manifest.jsonl"),
            "frozen_models_sha256": file_hash(models / "frozen-models.json"),
            "comparison_sha256": file_hash(models / "comparison.json"),
            "scope": "Actual full eligible corpus crop comparison; inspect per-source results and limitations",
        },
    )
    print("Corpus comparison completed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--beats-python", type=Path, required=True)
    parser.add_argument("--clap-python", type=Path, required=True)
    parser.add_argument("--beats-assets", type=Path, required=True)
    parser.add_argument("--clap-assets", type=Path, required=True)
    parser.add_argument("--urban-csv", type=Path, required=True)
    parser.add_argument("--attach-preparation", action="store_true")
    parser.add_argument("--attach-features", action="store_true")
    parser.add_argument("--attached-feature-pids", nargs="*", type=int, default=[])
    parser.add_argument("--feature-shards", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    os.environ.setdefault("PYTHONPATH", str(ROOT / "src"))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    run(args)


if __name__ == "__main__":
    main()
