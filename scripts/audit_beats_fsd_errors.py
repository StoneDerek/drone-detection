#!/usr/bin/env python3
"""Describe which publisher labels occur in paired BEATs false positives."""

import argparse
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path

from drone_detector.common import digest, file_hash, read_json, read_jsonl, write_json


def summarize(records, frozen, adapted):
    subsets = defaultdict(list)
    for i, row in enumerate(records):
        subsets["all"].append(i)
        stratum = "mechanical" if row["mechanical_confuser"] else "other"
        subsets[stratum].append(i)
        for tag in sorted(set(row["publisher_labels"])):
            subsets["label/" + tag].append(i)

    summary = {}
    for name, indices in subsets.items():
        groups = [records[i]["group_id"] for i in indices]
        new = [i for i in indices if adapted[i] and not frozen[i]]
        removed = [i for i in indices if frozen[i] and not adapted[i]]
        shared = [i for i in indices if frozen[i] and adapted[i]]
        new_groups = Counter(records[i]["group_id"] for i in new)
        summary[name] = dict(
            clips=len(indices),
            uploader_groups=len(set(groups)),
            frozen_false_positives=len(shared) + len(removed),
            adapted_false_positives=len(shared) + len(new),
            shared_false_positives=len(shared),
            new_false_positives=len(new),
            removed_false_positives=len(removed),
            new_false_positive_uploader_groups=len(new_groups),
            max_new_false_positives_from_one_uploader=max(new_groups.values(), default=0),
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "pair", "ledger", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    pair = read_json(args.pair)
    manifest_sha = file_hash(args.manifest)
    ledger_sha = file_hash(args.ledger)
    if (
        manifest_sha != pair["manifest_sha256"]
        or ledger_sha != pair["private_paired_ledger_sha256"]
    ):
        raise ValueError("Manifest or private ledger differs from the paired report")

    records = [r for r in read_jsonl(args.manifest) if r["source"] == "fsd50k-eval"]
    ledger = read_jsonl(args.ledger)
    background = {r["record_id_hash"]: r for r in ledger if r["source"] == "fsd50k-eval"}
    expected = [digest(r["id"]) for r in records]
    if (
        len(ledger) != pair["compared_validation_rows"]
        or len(ledger) != len({r["record_id_hash"] for r in ledger})
        or set(expected) != background.keys()
        or len(expected) != len(background)
    ):
        raise ValueError("The joined rows do not cover the complete paired background challenge")
    aligned = [background[digest(r["id"])] for r in records]
    if any(
        score["group_id_hash"] != digest(row["group_id"])
        or score["label"] != "non_drone"
        or row["label"] != "non_drone"
        or not row["publisher_labels"]
        or not isinstance(row["mechanical_confuser"], bool)
        for row, score in zip(records, aligned, strict=True)
    ):
        raise ValueError("Invalid joined groups or labels")

    frozen = [r["frozen_logit"] for r in aligned]
    adapted = [r["adapted_logit"] for r in aligned]
    if not all(map(math.isfinite, frozen + adapted)):
        raise ValueError("Nonfinite scores")
    frozen_locked = [s >= pair["frozen_cutoff_logit"] for s in frozen]
    adapted_locked = [s >= pair["adapted_cutoff_logit"] for s in adapted]
    if any(
        f != item["frozen_prediction"] or a != item["adapted_prediction"]
        for f, a, item in zip(frozen_locked, adapted_locked, aligned, strict=True)
    ):
        raise ValueError("Scores do not reproduce locked paired decisions")
    locked = summarize(records, frozen_locked, adapted_locked)
    observed = next(cell for cell in pair["source_class_pairs"] if cell["source"] == "fsd50k-eval")
    expected_counts = dict(
        frozen_false_positives=observed["frozen_errors"],
        adapted_false_positives=observed["adapted_errors"],
        shared_false_positives=observed["both_wrong"],
        new_false_positives=observed["new_errors_after_adaptation"],
        removed_false_positives=observed["corrected_by_adaptation"],
    )
    if any(locked["all"][name] != expected for name, expected in expected_counts.items()):
        raise ValueError("Locked counts differ from the paired result")

    cap = sum(frozen_locked)
    if cap >= len(adapted):
        raise ValueError("Frozen false-positive count leaves no background score at the cap")
    # Exclude the first score outside the budget and any score tied with it.
    boundary = sorted(adapted, reverse=True)[cap]
    matched = summarize(records, frozen_locked, [s > boundary for s in adapted])
    table = args.output.with_suffix(".csv")
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["comparison", "subset", *locked["all"]], lineterminator="\n"
        )
        writer.writeheader()
        for comparison, values in [("locked_dev", locked), ("matched_fsd", matched)]:
            for name, row in sorted(values.items()):
                writer.writerow(dict(comparison=comparison, subset=name, **row))
    strata = ("all", "mechanical", "other")
    write_json(
        args.output,
        dict(
            held_source=pair["held_source"],
            positive_mix=pair["positive_mix"],
            manifest_sha256=manifest_sha,
            paired_report_sha256=file_hash(args.pair),
            private_paired_ledger_sha256=ledger_sha,
            implementation_sha256=file_hash(__file__),
            locked_dev_cutoffs={name: locked[name] for name in strata},
            frozen_false_positive_cap=cap,
            adapted_cutoff_matched_to_frozen_fsd_cap={name: matched[name] for name in strata},
            label_table_sha256=file_hash(table),
            label_scope="Whole-recording publisher labels can co-occur and need not be audible in the scored crop.",
            comparison_scope="Exploratory description of saved scores on the development corpus.",
        ),
    )


if __name__ == "__main__":
    main()
