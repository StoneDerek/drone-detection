#!/usr/bin/env python3
"""Export the completed fixed comparison, numeric heads, plots and receipts."""

import argparse
import csv
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from drone_detector.common import file_hash, read_json, read_jsonl, write_json, write_jsonl

NAMES = {
    "original": "Original",
    "recalibrated": "Threshold only",
    "confuser-trained": "Confuser-trained head",
}


def percentage(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def run(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = args.report
    report.mkdir(parents=True, exist_ok=True)
    comparison = read_json(args.work / "models/final-comparison.json")
    completion = read_json(args.work / "prepared/preparation-complete.json")
    eligibility = [
        dict(id="fsd50k-eval-" + entry["id"], reason=entry["exclusion"])
        for part in (args.work / "prepared/parts").glob("*.json")
        for entry in read_json(part)["receipts"]
        if "exclusion" in entry
    ]
    completion["eligibility_exclusions"] = eligibility
    completion["exclusion_counts"] = dict(
        Counter(row["reason"] for row in eligibility + completion["exclusions"])
    )
    variants = comparison["variants"]
    rows = [r for r in read_jsonl(args.work / "prepared/manifest.jsonl") if r["split"] == "final"]
    lines = [
        "# BEATs mechanical-confuser results",
        "",
        "Three frozen comparisons were evaluated on the complete reserved cohort. "
        "BEATs remained pretrained and frozen; the third variant trained a new linear classification head.",
        "",
        f"Accepted fresh crops: **{len(rows):,}**. Candidate crops: **{completion['reserved_candidates']:,}**. "
        f"Quality/identity exclusions: **{completion['reserved_candidates'] - len(rows):,}**. "
        "The previous test is development evidence and is excluded from this table.",
        "",
        "| Fresh subset | Crops / provenance groups | Original | Threshold only | Confuser-trained head |",
        "|---|---:|---:|---:|---:|",
    ]
    metric_order = [
        ("fsd50k-eval", "FSD50K crop false-positive rate", "negative_clip_FPR"),
        ("fsd_mechanical", "FSD mechanical crop false-positive rate", "negative_clip_FPR"),
        ("wsu-air3", "WSU airborne crop recall", "positive_recall"),
        ("wsu-air3", "WSU ambient-tail crop false-positive rate", "negative_clip_FPR"),
        ("audrok-dresden", "AuDroK weak-positive crop recall", "positive_recall"),
    ]

    def subset(name, key):
        return variants[name]["subsets"].get(key) or variants[name]["metrics"]["by_source"][key]

    for key, label, metric in metric_order:
        m = subset("original", key)
        count = m["positive_clips"] if metric == "positive_recall" else m["negative_clips"]
        groups = m["provenance_groups"]
        if key == "wsu-air3" and metric == "negative_clip_FPR":
            groups = len(
                {r["group_id"] for r in rows if r["source"] == key and r["label"] == "non_drone"}
            )
        lines.append(
            f"| {label} | {count:,} / {groups:,} | "
            + " | ".join(percentage(subset(name, key)[metric]) for name in NAMES)
            + " |"
        )
    lines += [
        "",
        "Lower FPR and higher recall are better. Crop counts do not measure independent field trials.",
        "",
        "## What changed",
        "",
        "The threshold-only comparison uses the unchanged previous head. The third comparison adds "
        "6,089 UrbanSound8K negative crops to TRAIN, without adding positive training examples. "
        "Both new cutoffs follow the same broader-negative calibration rule. "
        "Their development operating points are not selected on these fresh recordings.",
        "",
        "| Variant | FSD FPR difference vs original | Paired uploader-cluster 95% interval |",
        "|---|---:|---:|",
    ]
    baseline = subset("original", "fsd50k-eval")["negative_clip_FPR"]
    for name in list(NAMES)[1:]:
        difference = subset(name, "fsd50k-eval")["negative_clip_FPR"] - baseline
        lo, hi = comparison["fsd_paired_clip_FPR_difference_vs_original_95pct_cluster_bootstrap"][
            name
        ]
        lines.append(
            f"| {NAMES[name]} | {100 * difference:+.2f} percentage points | {100 * lo:+.2f} to {100 * hi:+.2f} |"
        )
    lines += [
        "",
        "These intervals resample FSD uploader groups, not individual crops. They are conditional "
        "on the selected FSD collection. The few field campaigns do not support comparable population intervals.",
        "",
        "## Field variation",
        "",
        "| WSU flight | Positive / negative crops | Original recall / FPR | Threshold-only recall / FPR | New-head recall / FPR |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in sorted(variants["original"]["subsets"]):
        if not key.startswith("wsu_"):
            continue
        m = subset("original", key)
        lines.append(
            f"| {key.removeprefix('wsu_')} | {m['positive_clips']} / {m['negative_clips']} | "
            + " | ".join(
                percentage(subset(name, key)["positive_recall"])
                + " / "
                + percentage(subset(name, key)["negative_clip_FPR"])
                for name in NAMES
            )
            + " |"
        )
    lines += [
        "",
        "The two March flights share one date/site group. The ambient tail is from one recording; "
        "its190 windows are not190 independent negative recordings. AuDroK is one campaign. "
        "FSD labels identify human-annotated foreground sounds, not exhaustive drone absence. "
        "WSU airborne telemetry establishes physical presence, not audible drone onset/offset. "
        "AuDroK recording labels have not been independently verified for the central crops.",
        "",
        "## Evidence and reproducibility",
        "",
        "The complete cohort and all three numeric heads were frozen before fresh scoring. "
        "See [PROTOCOL.md](PROTOCOL.md), [metrics.csv](metrics.csv), [comparison.json](comparison.json), "
        "[data-quality.json](data-quality.json), and [error-cases.csv](error-cases.csv). "
        "Raw audio is not redistributed. Complete bounded acquisition receipts, the original reservation, "
        "canonical manifests and per-crop scores are retained in the local result bundle. "
        "The source checksum distinctions are documented in the protocol.",
        "",
        "This is a fixed crop-classification comparison. It does not establish operational false alarms per hour, "
        "event recall, detection delay, near-human performance, or independence from model pretraining data. "
        "The fresh cohort is consumed by this comparison; further tuning requires a separate evaluation design.",
        "",
        "![Fresh comparison](comparison.png)",
        "",
    ]
    (report / "RESULTS.md").write_text("\n".join(lines))
    write_json(report / "data-quality.json", completion)
    predictions = read_jsonl(args.work / "models/final-predictions.jsonl")
    by_id = {r["id"]: r for r in rows}
    with (report / "error-cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["id", "source", "label", "group_id", "reference", "labels", *NAMES.keys()],
        )
        writer.writeheader()
        for p in predictions:
            if not any(p[name]["predicted_drone"] != (p["label"] == "drone") for name in NAMES):
                continue
            r = by_id[p["id"]]
            writer.writerow(
                dict(
                    id=p["id"],
                    source=p["source"],
                    label=p["label"],
                    group_id=p["group_id"],
                    reference=r.get("attribution_url")
                    or r["native"].get("url")
                    or r["native"].get("member"),
                    labels=",".join(r.get("publisher_labels", [])),
                    **{name: int(p[name]["predicted_drone"]) for name in NAMES},
                )
            )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    colors = ["#6b7280", "#d97706", "#2563eb"]
    for ax, definitions, title in [
        (axes[0], metric_order[:2], "FSD negatives · lower is better"),
        (axes[1], [metric_order[3]], "WSU ambient tail · lower is better"),
        (axes[2], [metric_order[2], metric_order[4]], "Drone audio · higher is better"),
    ]:
        x = np.arange(len(definitions))
        for number, name in enumerate(NAMES):
            values = [100 * subset(name, key)[metric] for key, _, metric in definitions]
            bars = ax.bar(
                x + (number - 1) * 0.24, values, 0.23, label=NAMES[name], color=colors[number]
            )
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        ax.set_xticks(
            x,
            ["FSD all", "FSD mechanical"]
            if ax is axes[0]
            else ["190 windows / one recording"]
            if ax is axes[1]
            else ["WSU airborne", "AuDroK weak labels"],
        )
        ax.set_ylabel("Crop recall (%)" if ax is axes[2] else "Crop FPR (%)")
        ax.set_ylim(0, max(10, max(ax.get_ylim()[1] * 1.15, 105 if ax is axes[2] else 0)))
        ax.set_title(title, fontsize=11)
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Same frozen BEATs encoder · separate head and threshold effects", fontsize=13)
    figure.savefig(report / "comparison.png", dpi=170)
    plt.close(figure)
    bundle = args.work / "bundle"
    bundle.mkdir(exist_ok=True)
    for name in NAMES:
        destination = bundle / "models" / name
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ["head.npz", "model.json", "development.json"]:
            shutil.copyfile(args.work / "models" / name / filename, destination / filename)
    for filename in [
        "fit-plan.json",
        "frozen-models.json",
        "final-comparison.json",
        "final-predictions.jsonl",
        "final-use.json",
    ]:
        shutil.copyfile(args.work / "models" / filename, bundle / filename)
    shutil.copyfile(args.work / "plan.json", bundle / "reserved-cohort-plan.json")
    write_jsonl(
        bundle / "canonical-manifest.jsonl", read_jsonl(args.work / "prepared/manifest.jsonl")
    )
    for part in sorted((args.work / "prepared/parts").glob("*.json")):
        item = read_json(part)
        write_json(
            bundle / "source-receipts" / part.name,
            dict(
                plan_sha256=item["plan_sha256"],
                receipts=item["receipts"],
                records=[
                    {
                        k: v
                        for k, v in r.items()
                        if k in {"id", "native", "sha256", "decoded_sha256"}
                    }
                    for r in item["records"]
                ],
            ),
        )
    shutil.copytree(report, bundle / "report", dirs_exist_ok=True)
    (bundle / "report/bundle-receipt.json").unlink(missing_ok=True)
    archive = args.work / "confuser-study-v1-results.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(bundle))
    write_json(
        report / "bundle-receipt.json",
        dict(
            bytes=archive.stat().st_size,
            sha256=file_hash(archive),
            includes_raw_audio=False,
            includes_encoder_weights=False,
            includes_numeric_linear_heads=True,
        ),
    )
    print(
        json.dumps(
            dict(
                result_bundle=str(archive.resolve()),
                sha256=file_hash(archive),
                bytes=archive.stat().st_size,
            )
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=Path("work/confuser-study-v1"))
    p.add_argument("--report", type=Path, default=Path("reports/confuser-study-v1"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
