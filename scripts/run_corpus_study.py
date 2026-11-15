#!/usr/bin/env python3
"""Fit prospectively specified baselines, freeze decisions, then evaluate once."""

import argparse
import fcntl
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from drone_detector.common import digest, file_hash, read_json, write_json, write_jsonl
from drone_detector.data import load_manifest

FAMILIES = ["spectral", "clap", "beats", "level-duration"]


def weights(rows):
    sources = {
        label: {r["source"] for r in rows if r["label"] == label}
        for label in ["drone", "non_drone"]
    }
    groups = {
        (label, source): {
            r["group_id"] for r in rows if r["label"] == label and r["source"] == source
        }
        for label in sources
        for source in sources[label]
    }
    counts = Counter((r["label"], r["source"], r["group_id"]) for r in rows)
    if any(not values for values in sources.values()):
        raise ValueError("Both classes required")
    return np.asarray(
        [
            len(rows)
            / (
                2
                * len(sources[r["label"]])
                * len(groups[(r["label"], r["source"])])
                * counts[(r["label"], r["source"], r["group_id"])]
            )
            for r in rows
        ]
    )


def matrix(rows, cache, family):
    if family == "level-duration":
        result = []
        for row in rows:
            if file_hash(row["path"]) != row["sha256"]:
                raise ValueError("Canonical audio changed")
            audio, rate = sf.read(row["path"], dtype="float32")
            if rate != 16000 or audio.ndim != 1 or not np.isfinite(audio).all():
                raise ValueError("Invalid canonical input")
            rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
            peak = np.max(np.abs(audio))
            result.append(
                [
                    len(audio) / rate,
                    np.log10(max(rms, 1e-12)),
                    np.log10(max(peak, 1e-12)),
                    float(np.mean(np.abs(audio) < 1e-5)),
                ]
            )
        return np.asarray(result, dtype=np.float64)
    root = cache / family
    index = read_json(root / "index.json")
    vectors = []
    for row in rows:
        entry = index.get(row["id"])
        if entry is None:
            raise ValueError(f"Missing {family} embedding: {row['id']}")
        if entry["audio_sha256"] != row["sha256"]:
            raise ValueError("Audio/feature identity mismatch")
        path = root / (entry["key"] + ".npz")
        meta = read_json(path.with_suffix(".json"))
        if meta["specification"]["audio_sha256"] != row["sha256"]:
            raise ValueError("Feature receipt/audio identity mismatch")
        if meta["specification"]["encoder"] != read_json(root / "encoder.json"):
            raise ValueError("Feature receipt/encoder identity mismatch")
        if file_hash(path) != meta["sha256"]:
            raise ValueError("Corrupt feature payload")
        with np.load(path, allow_pickle=False) as saved:
            vectors.append(saved["vector"].copy())
    result = np.asarray(vectors, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite features")
    return result


def train(x, rows):
    y = np.asarray([int(r["label"] == "drone") for r in rows])
    mass = weights(rows)
    scaler = StandardScaler().fit(x, sample_weight=mass)
    with threadpool_limits(limits=1):
        model = LogisticRegression(C=1, max_iter=3000, tol=1e-6, random_state=42).fit(
            scaler.transform(x), y, sample_weight=mass
        )
    if model.n_iter_[0] >= 3000:
        raise ValueError("Classifier did not converge")
    return {
        "mean": scaler.mean_,
        "scale": scaler.scale_,
        "coef": model.coef_[0],
        "intercept": model.intercept_,
    }


def logits(x, head):
    return ((x - head["mean"]) / head["scale"]) @ head["coef"] + head["intercept"][0]


def threshold(scores, rows):
    negatives = np.sort(
        [s for s, r in zip(scores, rows, strict=True) if r["label"] == "non_drone"]
    )[::-1]
    if len(negatives) < 100:
        raise ValueError("Development negative sample too small for1% rule")
    allowed = int(np.floor(0.01 * len(negatives)))
    return float(np.nextafter(negatives[allowed], np.inf))


def metrics(rows, scores, cutoff):
    y = np.asarray([r["label"] == "drone" for r in rows], dtype=int)
    pred = np.asarray(scores) >= cutoff
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel()]
    recall = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    group_recalls = []
    group_fprs = []
    for group in {r["group_id"] for r in rows}:
        members = np.asarray([r["group_id"] == group for r in rows])
        if np.any(members & (y == 1)):
            group_recalls.append(float(np.mean(pred[members & (y == 1)])))
        if np.any(members & (y == 0)):
            group_fprs.append(float(np.mean(pred[members & (y == 0)])))
    return {
        "clips": len(rows),
        "provenance_groups": len({r["group_id"] for r in rows}),
        "positive_clips": tp + fn,
        "negative_clips": tn + fp,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "positive_recall": recall,
        "negative_clip_FPR": fpr,
        "macro_positive_group_recall": float(np.mean(group_recalls)) if group_recalls else None,
        "macro_negative_group_FPR": float(np.mean(group_fprs)) if group_fprs else None,
        "balanced_accuracy": (recall + 1 - fpr) / 2
        if recall is not None and fpr is not None
        else None,
        "roc_auc": float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, scores))
        if len(set(y)) == 2
        else None,
        "uncertainty_scope": "Descriptive held-out clip results; correlated source/campaign captures preclude treating clips as independent field trials",
    }


def summaries(rows, scores, cutoff):
    scores = np.asarray(scores)
    sources = sorted({r["source"] for r in rows})
    bysource = {}
    for source in sources:
        indices = [i for i, r in enumerate(rows) if r["source"] == source]
        bysource[source] = metrics([rows[i] for i in indices], scores[indices], cutoff)
    recalls = [m["positive_recall"] for m in bysource.values() if m["positive_recall"] is not None]
    fprs = [m["negative_clip_FPR"] for m in bysource.values() if m["negative_clip_FPR"] is not None]
    return {
        "pooled": metrics(rows, scores, cutoff),
        "by_source": bysource,
        "macro_positive_source_recall": float(np.mean(recalls)) if recalls else None,
        "macro_negative_source_FPR": float(np.mean(fprs)) if fprs else None,
        "macro_source_balanced_accuracy": float((np.mean(recalls) + 1 - np.mean(fprs)) / 2)
        if recalls and fprs
        else None,
    }


def subset_indices(rows, fraction, seed):
    selected = []
    for source in sorted({r["source"] for r in rows}):
        groups = sorted(
            {r["group_id"] for r in rows if r["source"] == source},
            key=lambda g: digest([seed, source, g]),
        )
        keep = set(groups[: max(1, int(np.ceil(len(groups) * fraction)))])
        selected.extend(
            i for i, r in enumerate(rows) if r["source"] == source and r["group_id"] in keep
        )
    return sorted(set(selected))


def freeze(args, families=None, authorize=True):
    if not (args.prepared / "preparation-complete.json").exists():
        raise ValueError("Complete corpus preparation first")
    completion = read_json(args.prepared / "preparation-complete.json")
    if completion["manifest_sha256"] != file_hash(args.prepared / "manifest.jsonl"):
        raise ValueError("Manifest changed after quality finalization")
    rows = load_manifest(args.prepared / "manifest.jsonl", require_split=True)
    training = [r for r in rows if r["split"] == "train"]
    development = [r for r in rows if r["split"] == "development"]
    plan = {
        "manifest_sha256": file_hash(args.prepared / "manifest.jsonl"),
        "protocol_sha256": file_hash(args.prepared / "protocol.json"),
        "models": FAMILIES,
        "classifier": "central-crop linear logistic C1; source/group/class balancing; TRAIN-only scaling",
        "threshold": "single global raw-logit cutoff from DEVELOPMENT negatives, empirical FPR<=1%; positive labels do not choose cutoff",
        "learning_curves": {
            "fractions": [0.25, 0.5, 1.0],
            "group_draw_seeds": [17, 43, 91],
            "evaluation": "DEVELOPMENT only",
        },
        "training_clips": len(training),
        "development_clips": len(development),
        "training_counts": dict(Counter(r["source"] for r in training)),
        "development_counts": dict(Counter(r["source"] for r in development)),
        "fitting_implementation_sha256": file_hash(__file__),
        "final_usage": "one prespecified batch after all heads and cutoffs are frozen",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output / "frozen-models.json"
    if (args.output / "fit-plan.json").exists() and read_json(
        args.output / "fit-plan.json"
    ) != plan:
        raise ValueError("Fit plan changed; preserve completed heads in this study directory")
    if frozen_path.exists() and not authorize:
        raise ValueError("Models already frozen; no further head fitting is permitted")
    write_json(args.output / "fit-plan.json", plan)
    results = {}
    for family in FAMILIES if families is None else families:
        path = args.output / family
        if (path / "model.json").exists() and (path / "development.json").exists():
            meta = read_json(path / "model.json")
            if meta["plan_sha256"] != file_hash(args.output / "fit-plan.json"):
                raise ValueError("Completed head belongs to a different fit plan")
            if file_hash(path / "head.npz") != meta["parameters_sha256"]:
                raise ValueError("Completed numeric head changed")
            if read_json(path / "development.json")["model"] != meta:
                raise ValueError("Completed development report identity changed")
            if meta["model_id"] != digest({k: v for k, v in meta.items() if k != "model_id"}):
                raise ValueError("Completed model metadata changed")
            if family != "level-duration" and meta["encoder"] != read_json(
                args.features / family / "encoder.json"
            ):
                raise ValueError("Completed head encoder changed")
            print("Reusing fitted", family, flush=True)
            results[family] = meta
            continue
        if frozen_path.exists():
            raise ValueError("A frozen head or its report is missing; do not refit it")
        print("Fitting", family, flush=True)
        x = matrix(training, args.features, family)
        dev = matrix(development, args.features, family)
        head = train(x, training)
        score = logits(dev, head)
        cutoff = threshold(score, development)
        path.mkdir(exist_ok=True)
        temporary = path / "head.partial.npz"
        np.savez_compressed(temporary, **head)
        temporary.replace(path / "head.npz")
        encoder = (
            read_json(args.features / family / "encoder.json")
            if family != "level-duration"
            else {
                "family": "level-duration-only",
                "input": "central-crop duration/logRMS/logpeak/nearzero fraction",
            }
        )
        meta = {
            "encoder": encoder,
            "cutoff_logit": cutoff,
            "probability_cutoff_display_only": float(expit(cutoff)),
            "parameters_sha256": file_hash(path / "head.npz"),
            "plan_sha256": file_hash(args.output / "fit-plan.json"),
            "feature_dimension": x.shape[1],
            "scope": "weak-label central-crop classification; sigmoid outputs are not calibrated field probabilities",
        }
        meta["model_id"] = digest(meta)
        write_json(path / "model.json", meta)
        curves = []
        for fraction in [0.25, 0.5, 1.0]:
            for seed in [17, 43, 91] if fraction < 1 else [17]:
                indices = subset_indices(training, fraction, seed)
                part = [training[i] for i in indices]
                candidate = head if fraction == 1 else train(x[indices], part)
                scores = logits(dev, candidate)
                decision = threshold(scores, development)
                curves.append(
                    {
                        "requested_group_fraction": fraction,
                        "selection_seed": seed,
                        "training_clips": len(part),
                        "training_groups": len({r["group_id"] for r in part}),
                        "counts": dict(Counter(r["source"] for r in part)),
                        "development": summaries(development, scores, decision),
                    }
                )
        development_report = {
            "model": meta,
            "metrics": summaries(development, score, cutoff),
            "learning_curves": curves,
        }
        write_json(path / "development.json", development_report)
        results[family] = meta
    if not authorize:
        print("Partial head fitting checkpointed; final evaluation remains locked.", flush=True)
        return
    if set(results) != set(FAMILIES):
        raise ValueError("Every prespecified model is required before final authorization")
    frozen = {"plan": plan, "models": results, "frozen_before_final_feature_extraction": True}
    if frozen_path.exists() and read_json(frozen_path) != frozen:
        raise ValueError("Previously frozen model set changed")
    write_json(frozen_path, frozen)
    write_json(
        args.prepared / "final-evaluation-authorized.json",
        {
            "frozen_models_sha256": file_hash(args.output / "frozen-models.json"),
            "model_ids": {name: data["model_id"] for name, data in results.items()},
        },
    )
    print("All heads and thresholds frozen; final feature extraction may begin.", flush=True)


def evaluate(args):
    rows = load_manifest(args.prepared / "manifest.jsonl", require_split=True)
    final = [r for r in rows if r["split"] == "final"]
    frozen = read_json(args.output / "frozen-models.json")
    authorization = read_json(args.prepared / "final-evaluation-authorized.json")
    if authorization["frozen_models_sha256"] != file_hash(args.output / "frozen-models.json"):
        raise ValueError("Frozen models changed after final authorization")
    if frozen["plan"]["manifest_sha256"] != file_hash(args.prepared / "manifest.jsonl"):
        raise ValueError("Manifest changed")
    if (args.output / "final-use.json").exists():
        raise ValueError("Final cohort already consumed; use saved results")
    for family in FAMILIES:
        if (
            family != "level-duration"
            and read_json(args.features / family / "encoder.json")
            != frozen["models"][family]["encoder"]
        ):
            raise ValueError("Final feature encoder differs from the frozen training encoder")
    # Require all final features before consuming the final cohort.
    matrices = {family: matrix(final, args.features, family) for family in FAMILIES}
    heads = {}
    for family in FAMILIES:
        path = args.output / family / "head.npz"
        if file_hash(path) != frozen["models"][family]["parameters_sha256"]:
            raise ValueError("Frozen head changed")
        with np.load(path, allow_pickle=False) as saved:
            heads[family] = {k: saved[k].copy() for k in saved.files}
    write_json(
        args.output / "final-use.json",
        {
            "frozen_models_sha256": file_hash(args.output / "frozen-models.json"),
            "manifest_sha256": file_hash(args.prepared / "manifest.jsonl"),
            "clips": len(final),
            "scope": "one prespecified final batch; subsequent tuning consumes this cohort",
        },
    )
    results = {}
    for family in FAMILIES:
        folder = args.output / family
        meta = frozen["models"][family]
        score = logits(matrices[family], heads[family])
        cutoff = meta["cutoff_logit"]
        pred = score >= cutoff
        result = {
            "model_id": meta["model_id"],
            "cutoff_logit": cutoff,
            "metrics": summaries(final, score, cutoff),
        }
        negative = [i for i, r in enumerate(final) if r["label"] == "non_drone"]
        pairs = {}
        for source in sorted({r["source"] for r in final if r["label"] == "drone"}):
            indices = negative + [i for i, r in enumerate(final) if r["source"] == source]
            pairs[source + " + heldout negatives"] = metrics(
                [final[i] for i in indices], score[indices], cutoff
            )
        result["paired_domain_metrics"] = pairs
        classes = {}
        for label in sorted({r.get("publisher_class") for r in final if r.get("publisher_class")}):
            indices = [i for i, r in enumerate(final) if r.get("publisher_class") == label]
            classes[label] = metrics([final[i] for i in indices], score[indices], cutoff)
        result["negative_classes"] = classes
        predictions = [
            {
                "id": r["id"],
                "source": r["source"],
                "group_id": r["group_id"],
                "truth": int(r["label"] == "drone"),
                "predicted": int(p),
                "logit": float(s),
                "score": float(expit(s)),
                "native": r["native"],
                "drone_id": r.get("drone_id"),
                "publisher_class": r.get("publisher_class"),
            }
            for r, p, s in zip(final, pred, score, strict=True)
        ]
        write_jsonl(folder / "final-predictions.jsonl", predictions)
        write_json(folder / "final.json", result)
        results[family] = result
    report = {
        "frozen_models_sha256": file_hash(args.output / "frozen-models.json"),
        "results": results,
        "scope": "Held-out source/capture central-crop weak-label classification; no operational reliability claim",
    }
    write_json(args.output / "comparison.json", report)
    print(json.dumps({name: r["metrics"] for name, r in results.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "freeze", "evaluate"])
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "fit" and args.family is None:
        parser.error("The fit stage requires one explicit --family")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".fitting.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "fit":
            freeze(args, [args.family], authorize=False)
        elif args.stage == "freeze":
            freeze(args)
        else:
            evaluate(args)


if __name__ == "__main__":
    main()
