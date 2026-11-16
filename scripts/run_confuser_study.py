#!/usr/bin/env python3
"""Freeze three BEATs comparisons; consume the reserved cohort once as a batch."""

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from drone_detector.common import digest, file_hash, read_json, write_json, write_jsonl
from drone_detector.data import load_manifest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_corpus_study import logits, matrix, metrics, summaries, threshold, weights

VARIANTS = ["original", "recalibrated", "confuser-trained"]


def broad_cutoff(scores, rows):
    """A single cutoff meeting the empirical1% rule on EACH negative source."""
    by_source = {}
    for source in sorted({r["source"] for r in rows if r["label"] == "non_drone"}):
        indices = [
            i for i, r in enumerate(rows) if r["source"] == source and r["label"] == "non_drone"
        ]
        by_source[source] = threshold(np.asarray(scores)[indices], [rows[i] for i in indices])
    return max(by_source.values()), by_source


def fit(args):
    models = args.work / "models"
    models.mkdir(exist_ok=True)
    if (models / "frozen-models.json").exists():
        raise ValueError("Heads and thresholds are already frozen")
    rows = load_manifest(args.work / "development-manifest.jsonl", require_split=True)
    source = args.previous / "models/beats"
    oldmeta = read_json(source / "model.json")
    if file_hash(source / "head.npz") != oldmeta["parameters_sha256"]:
        raise ValueError("Original BEATs head changed")
    if oldmeta["encoder"] != read_json(args.previous / "features/beats/encoder.json"):
        raise ValueError("Original embedding identity changed")
    original_training_mass = sum(r["previous_split"] == "train" for r in rows)
    plan = dict(
        reserved_cohort_sha256=file_hash(args.work / "plan.json"),
        development_manifest_sha256=file_hash(args.work / "development-manifest.jsonl"),
        implementation_sha256=file_hash(__file__),
        feature_implementation_sha256=file_hash(Path(__file__).with_name("run_corpus_study.py")),
        encoder=oldmeta["encoder"],
        original_model_id=oldmeta["model_id"],
        original_parameters_sha256=oldmeta["parameters_sha256"],
        variants=dict(
            original="unchanged previous head and cutoff",
            recalibrated="unchanged previous head; cutoff from broader negatives",
            confuser_trained="C1 linear head retrained with all Urban TRAIN parents; same broader-negative cutoff rule",
        ),
        training="same positive TRAIN, TUT and DNC; add deterministic75% Urban parents, stratified by publisher class",
        weights="equal class, source within class, parent/group within source; total weight mass fixed to original TRAIN size",
        weight_mass=original_training_mass,
        scaler="weighted TRAIN only",
        C=1.0,
        seed=42,
        max_iter=3000,
        tolerance=1e-6,
        cutoff="single global raw-logit cutoff =max of per-negative-source1% empirical cutoffs on DEVELOPMENT; positives do not choose cutoffs",
        evaluation="one reserved batch for all three models; original consumed FINAL is development only",
        counts=dict(Counter(r["split"] for r in rows)),
    )
    path = models / "fit-plan.json"
    if path.exists() and read_json(path) != plan:
        raise ValueError("Prospective fitting plan changed")
    write_json(path, plan)
    print("Loading cached TRAIN/DEV embeddings", len(rows), flush=True)
    x = matrix(rows, args.previous / "features", "beats")
    training = np.array([r["split"] == "train" for r in rows])
    development = ~training
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_rows = [r for r in rows if r["split"] == "development"]
    mass = weights(train_rows) * (original_training_mass / len(train_rows))
    scaler = StandardScaler().fit(x[training], sample_weight=mass)
    y = np.array([r["label"] == "drone" for r in train_rows], dtype=int)
    with threadpool_limits(limits=1):
        model = LogisticRegression(C=1, max_iter=3000, tol=1e-6, random_state=42).fit(
            scaler.transform(x[training]), y, sample_weight=mass
        )
    if model.n_iter_[0] >= 3000:
        raise ValueError("Head did not converge")
    newhead = dict(
        mean=scaler.mean_, scale=scaler.scale_, coef=model.coef_[0], intercept=model.intercept_
    )
    with np.load(source / "head.npz", allow_pickle=False) as saved:
        oldhead = {k: saved[k].copy() for k in saved.files}
    frozen = {}
    for name in VARIANTS:
        folder = models / name
        folder.mkdir(exist_ok=True)
        head = newhead if name == "confuser-trained" else oldhead
        if name == "confuser-trained":
            np.savez_compressed(folder / "head.npz", **head)
        else:
            shutil.copyfile(source / "head.npz", folder / "head.npz")
        scores = logits(x[development], head)
        if name == "original":
            cutoff, thresholds = (
                oldmeta["cutoff_logit"],
                {"original_TUT_only": oldmeta["cutoff_logit"]},
            )
        else:
            cutoff, thresholds = broad_cutoff(scores, dev_rows)
        meta = dict(
            name=name,
            parameters_sha256=file_hash(folder / "head.npz"),
            cutoff_logit=cutoff,
            component_cutoffs=thresholds,
            encoder=oldmeta["encoder"],
            fit_plan_sha256=file_hash(path),
            iterations=int(model.n_iter_[0]) if name == "confuser-trained" else None,
        )
        meta["model_id"] = digest(meta)
        write_json(folder / "model.json", meta)
        write_json(
            folder / "development.json",
            dict(
                metrics=summaries(dev_rows, scores, cutoff),
                by_urban_class={
                    category: metrics([dev_rows[i] for i in indices], scores[indices], cutoff)
                    for category in sorted(
                        {r["publisher_class"] for r in dev_rows if r["source"] == "urbansound8k"}
                    )
                    for indices in [
                        [i for i, r in enumerate(dev_rows) if r.get("publisher_class") == category]
                    ]
                },
            ),
        )
        frozen[name] = meta
        print(
            name,
            "cutoff",
            cutoff,
            "DEV",
            json.dumps(summaries(dev_rows, scores, cutoff)["pooled"]),
            flush=True,
        )
    write_json(models / "frozen-models.json", frozen)


def authorize(args):
    completion = read_json(args.work / "prepared/preparation-complete.json")
    frozen = read_json(args.work / "models/frozen-models.json")
    plan = read_json(args.work / "models/fit-plan.json")
    if file_hash(args.work / "plan.json") != plan["reserved_cohort_sha256"]:
        raise ValueError("Reserved cohort plan changed after fitting")
    if completion["manifest_sha256"] != file_hash(args.work / "prepared/manifest.jsonl"):
        raise ValueError("Prepared cohort changed")
    if set(frozen) != set(VARIANTS):
        raise ValueError("All three variants must be frozen")
    for name, meta in frozen.items():
        if file_hash(args.work / "models" / name / "head.npz") != meta["parameters_sha256"]:
            raise ValueError("Frozen parameters changed")
    write_json(
        args.work / "prepared/final-evaluation-authorized.json",
        dict(
            frozen_models_sha256=file_hash(args.work / "models/frozen-models.json"),
            manifest_sha256=completion["manifest_sha256"],
            reserved_cohort_sha256=file_hash(args.work / "plan.json"),
        ),
    )
    print("Authorized fixed-cohort embeddings and a single three-variant comparison", flush=True)


def evaluate(args):
    authorization = read_json(args.work / "prepared/final-evaluation-authorized.json")
    frozen_path = args.work / "models/frozen-models.json"
    if authorization["manifest_sha256"] != file_hash(
        args.work / "prepared/manifest.jsonl"
    ) or authorization["frozen_models_sha256"] != file_hash(frozen_path):
        raise ValueError("Final inputs no longer match the frozen batch")
    frozen = read_json(frozen_path)
    rows = [
        r
        for r in load_manifest(args.work / "prepared/manifest.jsonl", require_split=True)
        if r["split"] == "final"
    ]
    usage = dict(
        **authorization,
        evaluation_implementation_sha256=file_hash(__file__),
        scope="The full reserved cohort is consumed by this fixed three-variant batch",
    )
    use = args.work / "models/final-use.json"
    if use.exists() and read_json(use) != usage:
        raise ValueError("Fresh evaluation already consumed by a different batch")
    write_json(use, usage)
    if read_json(args.work / "features/beats/encoder.json") != frozen["original"]["encoder"]:
        raise ValueError("Final and training encoders differ")
    x = matrix(rows, args.work / "features", "beats")
    results = {}
    predictions = [
        dict(
            id=r["id"],
            source=r["source"],
            group_id=r["group_id"],
            label=r["label"],
            native=r["native"],
            mechanical_confuser=r.get("mechanical_confuser"),
            publisher_labels=r.get("publisher_labels"),
            capture_date=r.get("capture_date"),
            flight_id=r.get("flight_id"),
        )
        for r in rows
    ]
    all_scores = {}
    for name in VARIANTS:
        meta = frozen[name]
        head_path = args.work / "models" / name / "head.npz"
        if file_hash(head_path) != meta["parameters_sha256"]:
            raise ValueError("Frozen head changed")
        with np.load(head_path, allow_pickle=False) as saved:
            head = {k: saved[k].copy() for k in saved.files}
        scores = logits(x, head)
        cutoff = meta["cutoff_logit"]
        subsets = {
            "fsd_mechanical": [i for i, r in enumerate(rows) if r.get("mechanical_confuser")],
            **{
                "wsu_" + date + "_flight" + str(flight): [
                    i
                    for i, r in enumerate(rows)
                    if r.get("capture_date") == date and str(r.get("flight_id")) == str(flight)
                ]
                for date, flight in sorted(
                    {(r["capture_date"], r["flight_id"]) for r in rows if r["source"] == "wsu-air3"}
                )
            },
        }
        all_scores[name] = scores
        results[name] = dict(
            model=meta,
            metrics=summaries(rows, scores, cutoff),
            subsets={
                key: metrics([rows[i] for i in ids], scores[ids], cutoff)
                for key, ids in subsets.items()
            },
            fsd_by_label={
                label: metrics([rows[i] for i in ids], scores[ids], cutoff)
                for label in sorted({item for r in rows for item in r.get("publisher_labels", [])})
                for ids in [
                    [i for i, r in enumerate(rows) if label in r.get("publisher_labels", [])]
                ]
            },
        )
        for r, score in zip(predictions, scores, strict=True):
            r[name] = dict(logit=float(score), predicted_drone=bool(score >= cutoff))
    # Paired uploader-cluster bootstrap for FSD clip-FPR differences. Positive
    # field collections have too few campaigns for credible interval claims.
    fsd_groups = sorted({r["group_id"] for r in rows if r["source"] == "fsd50k-eval"})
    fp, totals = [], []
    for group in fsd_groups:
        ix = [
            i for i, r in enumerate(rows) if r["source"] == "fsd50k-eval" and r["group_id"] == group
        ]
        totals.append(len(ix))
        fp.append([sum(all_scores[name][ix] >= frozen[name]["cutoff_logit"]) for name in VARIANTS])
    fp, totals = np.asarray(fp), np.asarray(totals)
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(2000):
        draw = rng.integers(len(totals), size=len(totals))
        rates = fp[draw].sum(axis=0) / totals[draw].sum()
        differences.append(rates - rates[0])
    comparison = dict(
        variants=results,
        fsd_paired_clip_FPR_difference_vs_original_95pct_cluster_bootstrap={
            name: np.quantile(np.asarray(differences)[:, i], [0.025, 0.975]).tolist()
            for i, name in enumerate(VARIANTS)
        },
        uncertainty="FSD uploader-cluster resampling, conditional on this selected collection; field-campaign results descriptive",
        usage=usage,
    )
    write_json(args.work / "models/final-comparison.json", comparison)
    write_jsonl(args.work / "models/final-predictions.jsonl", predictions)
    args.report.mkdir(parents=True, exist_ok=True)
    with (args.report / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "variant",
                "subset",
                "clips",
                "positive_clips",
                "negative_clips",
                "tp",
                "fn",
                "fp",
                "tn",
                "recall",
                "FPR",
                "groups",
            ],
        )
        writer.writeheader()
        for name, value in results.items():
            for subset, m in {**value["metrics"]["by_source"], **value["subsets"]}.items():
                writer.writerow(
                    dict(
                        variant=name,
                        subset=subset,
                        clips=m["clips"],
                        positive_clips=m["positive_clips"],
                        negative_clips=m["negative_clips"],
                        **m["confusion"],
                        recall=m["positive_recall"],
                        FPR=m["negative_clip_FPR"],
                        groups=m["provenance_groups"],
                    )
                )
    write_json(args.report / "comparison.json", comparison)
    print(json.dumps({name: v["metrics"] for name, v in results.items()}, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["fit", "authorize", "evaluate"])
    p.add_argument("--work", type=Path, default=Path("work/confuser-study-v1"))
    p.add_argument("--previous", type=Path, default=Path("work/milestone-v1"))
    p.add_argument("--report", type=Path, default=Path("reports/confuser-study-v1"))
    args = p.parse_args()
    {"fit": fit, "authorize": authorize, "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
