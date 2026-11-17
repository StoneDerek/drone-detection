#!/usr/bin/env python3
"""Compare completed BEATs score rankings on the declared development folds."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from compare_completed_beats_arm import matrix
from report_beats_adaptation import FOLDS, MIXES, batches, verified_result
from run_corpus_study import logits

from drone_detector.adaptation import cohort_signature
from drone_detector.common import digest, file_hash, read_json, read_jsonl, write_json

FSD = "fsd50k-eval"
FSD_CAP_PERCENT = (0, 0.5, 1, 2, 5)


def counts(scores, rows, source, cutoff):
    truth = np.array([row["label"] == "drone" for row in rows if row["source"] == source])
    predictions = scores[[row["source"] == source for row in rows]] >= cutoff
    return dict(
        tp=int((truth & predictions).sum()),
        fn=int((truth & ~predictions).sum()),
        fp=int((~truth & predictions).sum()),
        tn=int((~truth & ~predictions).sum()),
    )


def ranking(scores, field_positive, fsd_negative, field_negative, fp_caps):
    ordered = np.sort(fsd_negative)[::-1]
    fsd_truth = np.r_[np.ones(len(field_positive)), np.zeros(len(fsd_negative))]
    comparisons = []
    for cap in sorted(fp_caps):
        # A tied score is kept out as a block if including it would exceed the cap.
        cutoff = float(np.nextafter(ordered[cap], np.inf))
        comparisons.append(
            dict(
                allowed_fsd_false_positives=cap,
                observed_fsd_false_positives=int((fsd_negative >= cutoff).sum()),
                detected_field_positives=int((field_positive >= cutoff).sum()),
                field_background_false_positives=(
                    int((field_negative >= cutoff).sum()) if len(field_negative) else None
                ),
            )
        )
    return dict(
        validation_score_sha256=digest(scores.tolist()),
        field_positive_vs_fsd_auc=float(
            roc_auc_score(fsd_truth, np.r_[field_positive, fsd_negative])
        ),
        field_positive_vs_local_background_auc=(
            float(
                roc_auc_score(
                    np.r_[np.ones(len(field_positive)), np.zeros(len(field_negative))],
                    np.r_[field_positive, field_negative],
                )
            )
            if len(field_negative)
            else None
        ),
        equal_fsd_caps=comparisons,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.work
    rows = read_jsonl(work / "manifest.jsonl")
    plan = read_json(work / "plan.json")
    plan_sha = file_hash(work / "plan.json")
    if cohort_signature(rows) != plan["recipe"]["cohort_signature"]:
        raise ValueError("Study cohort changed")
    recipe = plan["recipe"]["fine_tuning"]
    lookup = {row["id"]: i for i, row in enumerate(rows)}
    base, base_meta = matrix(work / "base-embeddings.npz", rows)
    report = {}
    for fold in FOLDS:
        split = plan["folds"][fold]
        indices = [lookup[i] for i in split["validation"]]
        validation = [rows[i] for i in indices]
        field_positive = np.array(
            [r["source"] == fold and r["label"] == "drone" for r in validation]
        )
        field_negative = np.array(
            [r["source"] == fold and r["label"] == "non_drone" for r in validation]
        )
        fsd_negative = np.array(
            [r["source"] == FSD and r["label"] == "non_drone" for r in validation]
        )
        if not np.all(field_positive | field_negative | fsd_negative) or not (
            field_positive.any() and fsd_negative.any()
        ):
            raise ValueError("Unexpected validation source or label")
        models = {}
        observed_caps = set()
        for mix in MIXES:
            ids = split[mix + "_positive_train"]
            steps = batches(rows, ids, recipe["batch_size"]) * recipe["epochs"]
            for family in ("frozen", "fine-tuned"):
                result = verified_result(
                    work, fold, mix, family, plan_sha, steps, len(ids), digest(ids), recipe
                )
                if result is None:
                    if family == "frozen":
                        raise ValueError("Missing matched frozen control")
                    continue
                folder = work / fold / mix / family
                model = result["model"]
                if family == "frozen":
                    x, meta = base, base_meta
                    if meta["encoder"] != model["encoder"]:
                        raise ValueError("Frozen matrix does not match the classifier")
                else:
                    x, meta = matrix(folder / "embeddings.npz", rows)
                    if meta["adapted_parameters_sha256"] != model["selected_encoder_sha256"]:
                        raise ValueError("Adapted matrix does not match the classifier")
                with np.load(folder / "head.npz", allow_pickle=False) as head:
                    scores = logits(x[indices], head)
                if not np.isfinite(scores).all():
                    raise ValueError("Nonfinite validation scores")
                locked = {
                    source: counts(scores, validation, source, model["cutoff_logit"])
                    for source in (fold, FSD)
                }
                if locked != {
                    s: v["confusion"] for s, v in result["validation"]["by_source"].items()
                }:
                    raise ValueError("Reconstructed decisions do not match completed receipts")
                observed_caps.add(locked[FSD]["fp"])
                models[mix + "/" + family] = (
                    scores,
                    dict(
                        locked_cutoff_counts=locked,
                        classifier_sha256=model["parameters_sha256"],
                        matrix_sha256=meta["sha256"],
                    ),
                )
        fsd_n = int(fsd_negative.sum())
        caps = {int(percent * fsd_n / 100) for percent in FSD_CAP_PERCENT}
        caps.update(cap for cap in observed_caps if cap < fsd_n)
        report[fold] = dict(
            validation_rows=len(validation),
            field_positive_clips=int(field_positive.sum()),
            field_background_clips=int(field_negative.sum()),
            fsd_negative_clips=fsd_n,
            models={
                name: dict(
                    meta,
                    **ranking(
                        scores,
                        scores[field_positive],
                        scores[fsd_negative],
                        scores[field_negative],
                        caps,
                    ),
                )
                for name, (scores, meta) in models.items()
            },
        )
    write_json(
        args.output,
        dict(
            plan_sha256=plan_sha,
            manifest_sha256=file_hash(work / "manifest.jsonl"),
            implementation_sha256=file_hash(__file__),
            fsd_cap_percent=FSD_CAP_PERCENT,
            method="Post-hoc development ranking comparison. Caps use held FSD scores; model thresholds stay locked.",
            folds=report,
        ),
    )


if __name__ == "__main__":
    main()
