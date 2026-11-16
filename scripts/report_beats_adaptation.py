#!/usr/bin/env python3
"""Describe actual completed development controls and pending adaptation work."""

import argparse
import csv
import shutil
from pathlib import Path

from drone_detector.adaptation import partition_summary
from drone_detector.common import read_json, read_jsonl, write_json


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=Path("work/beats-adaptation-v1"))
    p.add_argument("--output", type=Path, default=Path("reports/beats-adaptation-v1"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = read_json(args.work / "frozen-controls.json")
    plan = read_json(args.work / "plan.json")
    rows = read_jsonl(args.work / "manifest.jsonl")
    lines = [
        "# Positive-diversity development controls",
        "",
        "The four frozen-encoder controls are complete. Full BEATs fine-tuning is a separate running/queued phase; "
        "the results below do not claim that the encoder has been task-trained.",
        "",
        "These are **known-corpus development comparisons**, not fresh generalization tests. "
        "Each fold excludes its entire field source from fitting and model selection. "
        "Negative training and selection are identical across the two positive mixtures within a fold.",
        "",
        "| Held field source | Positive TRAIN mix | Positive crops / groups | FSD crop FPR | Held-field drone recall | Held-field background FPR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    values = []
    for fold in ["wsu-air3", "audrok-dresden"]:
        for mix in ["current", "broader"]:
            result = frozen[fold][mix]
            training = partition_summary(rows, plan["folds"][fold][mix + "_positive_train"])
            positive_count = sum(
                n for k, n in training["by_source_label"].items() if k.endswith("/drone")
            )
            field = result["validation"]["by_source"][fold]
            fsd = result["validation"]["by_source"]["fsd50k-eval"]
            lines.append(
                f"| {fold} | {mix} | {positive_count:,} / {training['positive_groups']} | "
                f"{pct(fsd['negative_clip_FPR'])} | {pct(field['positive_recall'])} | {pct(field['negative_clip_FPR'])} |"
            )
            values.append(
                dict(
                    held_source=fold,
                    positive_mix=mix,
                    positive_training_clips=positive_count,
                    positive_training_groups=training["positive_groups"],
                    FSD_FP=fsd["confusion"]["fp"],
                    FSD_N=fsd["negative_clips"],
                    field_TP=field["confusion"]["tp"],
                    field_P=field["positive_clips"],
                    field_FP=field["confusion"]["fp"],
                    field_N=field["negative_clips"],
                )
            )
    lines += [
        "",
        "Broader positive training does not consistently help in this frozen representation. "
        "Adding AuDroK positives modestly reduces WSU background errors, with a small recall loss. "
        "Adding WSU positives lowers AuDroK recall and increases FSD false positives. "
        "This does not establish whether weak labels, source coverage, learned features or another factor causes the failures.",
        "",
        "For the AuDroK-held fold, both mixtures include190 WSU negative windows in TRAIN. "
        "The broader mixture also includes WSU positives. For the WSU-held fold, AuDroK contributes only additional positives. "
        "WSU background and positive crops from the same capture are never split between fitting and held-field validation.",
        "",
        "The current positive TRAIN has976 crops from Poland and Berne, spanning four groups. "
        "The broader mixtures add either58 AuDroK crops from one campaign, or1,486 WSU positive windows from three date/site groups. "
        "The7,931 FSD recordings are reused across the two folds; they are not15,862 independent negatives. "
        "WSU's190 negative windows come from one recording. Labels remain weak publisher labels; audibility review is pending.",
        "",
        "## Full encoder comparison",
        "",
        "The next four arms use all90,311,792 BEATs representation parameters, three complete epochs, batch size8, "
        "and the unchanged native frontend. Each adapted model starts from its paired trained C1 head. "
        "The best trained epoch is selected using weighted selection-DEV log-loss, then the same C1 head "
        "is refitted on TRAIN embeddings and the same global calibration rule is applied. "
        "No held-field or FSD score selects the checkpoint or cutoff.",
        "",
        "Training recovery stores model, optimizer, CPU/CUDA/Python/NumPy RNG and the exact next batch. "
        "Checksums and pending receipts protect interrupted checkpoint publication. Every selected training crop is visited "
        "once per epoch; exact-length batches avoid adding waveform padding. A44-hour cumulative pipeline budget "
        "requests a cooperative checkpoint pause rather than reducing the dataset or epoch count.",
        "",
        "The GPU notebook reconstructs the full corpus from existing pinned originals and public field sources. "
        "It creates its own base embeddings and does not require a previous feature cache. "
        "Canonical audio and checkpoints persist in a separate private Drive project; verified local copies feed training.",
        "",
        "The existing blind field-review inventory remains useful while training runs. Full-backbone updates and "
        "training/recovery mechanisms have been checked on synthetic signals; those software checks are not detector performance evidence.",
        "",
    ]
    text = "\n".join(lines)
    for old, new in [
        ("include190", "include 190"),
        ("has976", "has 976"),
        ("either58", "either 58"),
        ("or1,486", "or 1,486"),
        ("The7,931", "The 7,931"),
        ("not15,862", "not 15,862"),
        ("WSU's190", "WSU's 190"),
        ("all90,311", "all 90,311"),
        ("size8", "size 8"),
        ("A44-hour", "A 44-hour"),
    ]:
        text = text.replace(old, new)
    (args.output / "RESULTS.md").write_text(text)
    with (args.output / "frozen-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    write_json(args.output / "frozen-controls.json", frozen)
    for fold in frozen:
        for mix in frozen[fold]:
            source = args.work / fold / mix / "frozen"
            destination = args.output / "heads" / fold / mix
            destination.mkdir(parents=True, exist_ok=True)
            for name in ["head.npz", "model.json"]:
                shutil.copyfile(source / name, destination / name)
    print(args.output / "RESULTS.md")


if __name__ == "__main__":
    main()
