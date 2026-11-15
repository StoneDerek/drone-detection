#!/usr/bin/env python3
"""Produce an inspectable report from a completed frozen comparison, without fitting."""

import argparse
import csv
import io
from collections import defaultdict
from pathlib import Path

import numpy as np

from drone_detector.common import file_hash, read_json, read_jsonl, write_json
from drone_detector.data import load_manifest

NAMES = {
    "spectral": "Spectral summaries",
    "clap": "Frozen CLAP",
    "beats": "Frozen BEATs",
    "level-duration": "Level/duration control",
}


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def source_hours(rows):
    """Union documented excerpt offsets; other sources retain file-hour wording."""
    intervals = defaultdict(list)
    for row in rows:
        source = row.get("publisher_source_interval")
        if source is None:
            continue
        native = row["native"]
        a = source[0] + native["crop_start_frame"] / native["sample_rate"]
        b = source[0] + native["crop_end_frame"] / native["sample_rate"]
        intervals[row["original_id"]].append((a, b))
    if not intervals:
        return None
    elapsed = 0.0
    for group in intervals.values():
        end = -float("inf")
        for a, b in sorted(group):
            elapsed += max(0, b - max(a, end))
            end = max(end, b)
    return elapsed / 3600


def report(study, output):
    output.mkdir(parents=True, exist_ok=True)
    prepared = study / "prepared"
    models = study / "models"
    rows = load_manifest(prepared / "manifest.jsonl", require_split=True)
    comparison = read_json(models / "comparison.json")
    frozen = read_json(models / "frozen-models.json")
    quality = read_json(prepared / "quality-finalization.json")
    if comparison["frozen_models_sha256"] != file_hash(models / "frozen-models.json"):
        raise ValueError("Report model identity changed")
    if frozen["plan"]["manifest_sha256"] != file_hash(prepared / "manifest.jsonl"):
        raise ValueError("Report cohort changed")
    results = comparison["results"]
    predictions = {
        family: {r["id"]: r for r in read_jsonl(models / family / "final-predictions.jsonl")}
        for family in NAMES
    }
    final = {r["id"]: r for r in rows if r["split"] == "final"}
    if any(set(items) != set(final) for items in predictions.values()):
        raise ValueError("Prediction coverage does not match the final cohort")
    csv_rows = []
    lines = [
        "# Public-corpus frozen-representation comparison",
        "",
        "This is an executed comparison of fresh linear classifiers on real public audio. "
        "Every eligible file contributes a prospectively fixed central crop of at most three seconds. "
        "The result measures transfer of weak recording labels to these crops. It does not measure "
        "continuous-stream event recall, false alarms per hour, alarm occupancy, or detection delay.",
        "",
        f"**{len(rows):,} accepted crops**, "
        f"{sum(r['duration_seconds'] for r in rows) / 3600:.3f} selected crop-hours, "
        f"from {sum(r['native']['duration_seconds'] for r in rows) / 3600:.3f} summed native-file hours. "
        "These hour totals are not independent drone-audible or drone-absent exposure.",
        "",
        "## Cohort and partition",
        "",
        "| Release / partition | Crops | Provenance groups | Native-file h | Selected crop h | Documented crop-interval union h |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    coverage = []
    for source, split in sorted({(r["source"], r["split"]) for r in rows}):
        selected = [r for r in rows if r["source"] == source and r["split"] == split]
        union = source_hours(selected)
        facts = {
            "source": source,
            "split": split,
            "crops": len(selected),
            "groups": len({r["group_id"] for r in selected}),
            "native_file_hours": sum(r["native"]["duration_seconds"] for r in selected) / 3600,
            "selected_crop_hours": sum(r["duration_seconds"] for r in selected) / 3600,
            "documented_crop_interval_union_hours": union,
        }
        coverage.append(facts)
        lines.append(
            f"| {source} / {split} | {facts['crops']:,} | {facts['groups']} | "
            f"{facts['native_file_hours']:.3f} | {facts['selected_crop_hours']:.3f} | "
            + (f"{union:.3f}" if union is not None else "not established")
            + " |"
        )
    lines += [
        "",
        "TRAIN uses three Polish capture dates, the single Berne campaign, TUT2017 development "
        "and DNC. DEVELOPMENT uses a fourth Polish date and TUT2017 evaluation. FINAL uses the "
        "fifth Polish date at a new site, Edzell DroneNoise, processed NEAPTIDE measurements, "
        "and UrbanSound8K. Simultaneous microphones are not counted as independent examples: "
        "channel 0 is used for multichannel WAVs, microphone M1 for DroneNoise, and microphone 1 "
        "for NEAPTIDE. DNC has unresolved session provenance and is conservatively one group.",
        "",
        "The final positive sources each represent one campaign/date; thousands of files cannot "
        "turn them into thousands of independent domain-transfer trials. Source and class remain "
        "partly confounded. Positive crops inherit publisher drone-recording labels without "
        "independent audibility timing. Polish speech-plus-drone files remain positive. General "
        "sound-class labels do not certify exhaustive drone absence. NEAPTIDE is a processed "
        "measurement stress set, not unmodified field audio.",
        "",
        f"Quality finalization made {len(quality['additional_exclusions'])} additional exclusions "
        "for publisher fault intervals, verified duplicates, or historical parent overlap. "
        "UrbanSound8K parents shared with the project's earlier ESC-50 material are quarantined "
        "before final scoring. Shared parent identity does not assert identical selected excerpts. "
        "Source acquisition checks publisher "
        "checksums, stored SHA-256, and selected ZIP-member CRC. TUT original IDs are disjoint "
        "across releases. Shifted/lossy cross-source copies and overlap in encoder pretraining "
        "have not been ruled out.",
        "",
        "## Frozen held-out results",
        "",
        "Every head uses weighted TRAIN-only standardization and logistic regression with fixed "
        "C=1. Weight mass is balanced by class, source, capture group and then rows. One raw-logit "
        "cutoff per model is chosen from DEVELOPMENT negatives at empirical clip FPR ≤1%, "
        "then retained for every final source. No final-source threshold is fitted. Sigmoid values "
        "are display scores, not calibrated field probabilities.",
        "",
        "| Model | Polish new-site recall | Edzell recall | NEAPTIDE recall | Urban clip FPR | Urban parent-mean FPR | Macro-source balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, label in NAMES.items():
        result = results[family]
        metrics = result["metrics"]
        bysource = metrics["by_source"]
        values = [
            bysource[s]["positive_recall"]
            for s in ["poland-uav-v1", "dronenoise-edzell-v3", "neaptide-v1"]
        ]
        values += [
            bysource["urbansound8k"]["negative_clip_FPR"],
            bysource["urbansound8k"]["macro_negative_group_FPR"],
            metrics["macro_source_balanced_accuracy"],
        ]
        lines.append("| " + label + " | " + " | ".join(percent(v) for v in values) + " |")
        for source, values in bysource.items():
            csv_rows.append(
                {
                    "model": family,
                    "source": source,
                    **{
                        k: values[k]
                        for k in [
                            "clips",
                            "provenance_groups",
                            "positive_recall",
                            "negative_clip_FPR",
                            "macro_negative_group_FPR",
                        ]
                    },
                }
            )
    lines += [
        "",
        "Macro-source balanced accuracy gives equal weight to the three positive-source recalls, "
        "then balances that mean against negative-source specificity. The pooled score is retained "
        "in JSON but does not let the large UrbanSound8K excerpt count dominate the comparison.",
        "Urban parent-mean FPR averages each original recording's crop-level FPR with equal "
        "weight, so parents supplying many overlapping excerpts do not receive more mass. "
        "This also remains a crop statistic, rather than an alarm-episode rate.",
        "",
        "AUC distinguishes score ranking from the behavior of the development-selected cutoff. "
        "Each row below pairs one final positive source with the same held-out UrbanSound8K "
        "negative set; these rows share negatives and are not independent experiments.",
        "",
        "| Model | Polish + Urban AUC | Edzell + Urban AUC | NEAPTIDE + Urban AUC |",
        "|---|---:|---:|---:|",
    ]
    for family, label in NAMES.items():
        pairs = results[family]["paired_domain_metrics"]
        values = [
            pairs[s + " + heldout negatives"]["roc_auc"]
            for s in ["poland-uav-v1", "dronenoise-edzell-v3", "neaptide-v1"]
        ]
        lines.append("| " + label + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += [
        "",
        "## False positives by published negative class",
        "",
        "| UrbanSound8K class | Spectral FPR | CLAP FPR | BEATs FPR | Level/duration FPR |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in sorted(results["spectral"]["negative_classes"]):
        values = [results[f]["negative_classes"][label]["negative_clip_FPR"] for f in NAMES]
        lines.append("| " + label + " | " + " | ".join(percent(v) for v in values) + " |")
    lines += [
        "",
        "## Polish measurement conditions",
        "",
        "These are subdivisions of the same final capture date, not additional independent "
        "domains. Speech markers come from publisher filenames; they do not constitute a new "
        "audibility annotation. All decisions below use the frozen global cutoffs.",
        "",
        "| Condition | Crops | Spectral recall | CLAP recall | BEATs recall | Level/duration recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    polish = [r for r in final.values() if r["source"] == "poland-uav-v1"]
    conditions = [
        ("Publisher speech marker", [r for r in polish if r.get("speech_marker")]),
        ("Without speech marker", [r for r in polish if not r.get("speech_marker")]),
    ]
    conditions += [
        (identifier, [r for r in polish if r.get("drone_id") == identifier])
        for identifier in sorted({r["drone_id"] for r in polish})
    ]
    for label, selected in conditions:
        values = [
            float(np.mean([predictions[family][r["id"]]["predicted"] for r in selected]))
            for family in NAMES
        ]
        lines.append(
            f"| {label} | {len(selected)} | " + " | ".join(percent(v) for v in values) + " |"
        )
    lines += [
        "",
        "## Development-only grouped learning curves",
        "",
        "These fits use 25%, 50% and 100% of available groups within each training source. "
        "Partial fits repeat group selection with seeds 17, 43 and 91. Single-group sources "
        "remain present at every size; this is not an estimate of a universal required "
        "training-hour count. No final examples choose these subsets or their cutoffs.",
        "",
        "| Model | Requested group fraction | Fits | Training groups | Training clips range | Development recall mean (range) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family, label in NAMES.items():
        development = read_json(models / family / "development.json")
        for fraction in [0.25, 0.5, 1.0]:
            subset = [
                r
                for r in development["learning_curves"]
                if r["requested_group_fraction"] == fraction
            ]
            recalls = [r["development"]["macro_positive_source_recall"] for r in subset]
            counts = [r["training_clips"] for r in subset]
            groups = [r["training_groups"] for r in subset]
            lines.append(
                f"| {label} | {fraction:.0%} | {len(subset)} | {min(groups)}–{max(groups)} | {min(counts):,}–{max(counts):,} | "
                f"{percent(float(np.mean(recalls)))} ({percent(min(recalls))}–{percent(max(recalls))}) |"
            )
    lines += [
        "",
        "## Interpretation limits and next use",
        "",
        "The primary comparison is pretrained representations versus a fresh spectral head on "
        "the same corpus. The nuisance control asks how much simple level/duration cues can "
        "predict across this partition; it does not establish a causal explanation. A lower "
        "false-positive rate with weaker recall, or higher recall with widespread false positives, "
        "is not automatically an improvement. Inspect the full per-source table and paired AUCs.",
        "",
        "All four prespecified models were evaluated together after their heads and thresholds "
        "were frozen. This final cohort is now consumed. Tuning to these results requires a "
        "different independent evaluation cohort for a new generalization claim. Conditional "
        "clip metrics are reported without confidence intervals that pretend correlated files "
        "are independent campaigns. A useful operational detector still needs matched field "
        "backgrounds and temporal annotations supporting audible-event and false-alarm claims.",
        "",
        "## Reproduction and evidence",
        "",
        f"- Final manifest SHA-256: `{file_hash(prepared / 'manifest.jsonl')}`",
        f"- Frozen models SHA-256: `{file_hash(models / 'frozen-models.json')}`",
        f"- Comparison SHA-256: `{file_hash(models / 'comparison.json')}`",
        "- Encoder checkpoint, frontend, package, code and feature identities are in the machine-readable receipts.",
        "- `scripts/run_corpus_milestone.py` executes preparation, quality finalization, frozen fitting and one final evaluation.",
        "- `scripts/report_corpus_study.py` generates this report from saved results without refitting or rescoring audio.",
        "- Source licenses remain per release; this report does not redistribute the audio or change its terms.",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines))
    fields = [
        "model",
        "source",
        "clips",
        "provenance_groups",
        "positive_recall",
        "negative_clip_FPR",
        "macro_negative_group_FPR",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(csv_rows)
    (output / "metrics.csv").write_text(buffer.getvalue())
    write_json(output / "coverage.json", coverage)
    write_json(output / "comparison.json", comparison)
    write_json(
        output / "quality-summary.json",
        {k: v for k, v in quality.items() if k != "verified_gain_duplicate_pairs"},
    )
    case_fields = [
        "id",
        "source",
        "truth",
        "group_id",
        "native_member",
        "publisher_class",
        "drone_id",
        "speech_plus_drone",
        "models_wrong",
    ]
    case_fields += [family + suffix for family in NAMES for suffix in ["_logit", "_decision"]]
    with (output / "error-review.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=case_fields)
        writer.writeheader()
        for identifier, row in final.items():
            truth = int(row["label"] == "drone")
            wrong = [
                family for family in NAMES if predictions[family][identifier]["predicted"] != truth
            ]
            if not wrong:
                continue
            case = {
                "id": identifier,
                "source": row["source"],
                "truth": truth,
                "group_id": row["group_id"],
                "native_member": row["native"]["member"],
                "publisher_class": row.get("publisher_class"),
                "drone_id": row.get("drone_id"),
                "speech_plus_drone": row.get("speech_marker"),
                "models_wrong": ";".join(wrong),
            }
            for family in NAMES:
                case[family + "_logit"] = predictions[family][identifier]["logit"]
                case[family + "_decision"] = predictions[family][identifier]["predicted"]
            writer.writerow(case)
    for family in NAMES:
        write_json(
            output / (family + "-development.json"), read_json(models / family / "development.json")
        )
    print(output / "RESULTS.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.study, args.output)


if __name__ == "__main__":
    main()
