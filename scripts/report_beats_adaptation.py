#!/usr/bin/env python3
"""Report completed source-held-out controls and verified execution progress."""

import argparse
import csv
import datetime
import json
import shutil
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

from drone_detector.adaptation import cohort_signature, partition_summary
from drone_detector.common import digest, file_hash, read_json, read_jsonl, write_json

FOLDS = ["wsu-air3", "audrok-dresden"]
MIXES = ["current", "broader"]


def pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def process_active(pid, work):
    if not isinstance(pid, int):
        return False
    root = Path(f"/proc/{pid}")
    try:
        # PIDs can be reused after a runtime replacement. Match exact arguments.
        arguments = (root / "cmdline").read_bytes().split(b"\0")
        state = (root / "stat").read_text().rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, PermissionError):
        return False
    return state != "Z" and str(work.resolve()).encode() in arguments and any(
        value.endswith(b"/scripts/train_beats_adaptation.py")
        or value.endswith(b"/scripts/coordinate_beats_arms.py")
        or value.endswith(b"/scripts/run_beats_adaptation.py")
        for value in arguments
    )


def execution(work, total_batches):
    path = work / "pipeline-state.json"
    if not path.exists():
        return dict(status="not_started", completed_training_arms=0)
    state = read_json(path)
    active = process_active(state.get("child_pid"), work)
    status = state["status"]
    if status in {"running", "pausing"} and not active:
        status = "interrupted"
    result = dict(
        status=status,
        saved_supervisor_status=state["status"],
        active_child_verified=active,
        stage=state.get("stage"),
        accumulated_hours=state["accumulated_seconds"] / 3600,
        max_wall_hours=state.get("max_wall_hours", 44),
        checkpoint_every_steps=state["identity"].get("checkpoint_every_steps", 1000),
        device=state["identity"]["device"],
        threads=state["identity"]["threads"],
    )
    log = work / "four-full-fine-tuning-arms.log"
    if log.exists():
        first_segment, latest, highest, resumed = [], None, 0, None
        first_segment_ended = False
        for line in log.read_text().splitlines():
            if line.startswith(("Completed full epoch ", "Completed full adaptation arm ")):
                first_segment_ended = True
            if line.startswith("Resuming exact training batch "):
                first_segment_ended = True
                resumed = int(line.split()[-1])
                latest = None
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict) or "batch" not in record or "steps" not in record:
                continue
            latest = record
            highest = max(highest, record["steps"])
            if not first_segment_ended:
                first_segment.append(record)
        result["last_logged_batch"] = latest
        result["maximum_logged_steps_in_an_arm"] = highest
        result["last_resume_checkpoint_steps"] = resumed
        if first_segment:
            last = first_segment[-1]
            seconds = last["elapsed_seconds"] / last["steps"]
            result["first_training_segment_timing"] = dict(
                observed_steps=last["steps"],
                elapsed_seconds=last["elapsed_seconds"],
                seconds_per_batch=seconds,
                planned_total_training_batches=total_batches,
                projected_training_only_hours=seconds * total_batches / 3600,
                scope="Extrapolation from actual first-arm batches; excludes selection, feature extraction, replay and IO variability",
            )
    return result


def batches(rows, ids, batch_size):
    selected = set(ids)
    lengths = Counter(
        round(r["duration_seconds"] * 16000) for r in rows if r["id"] in selected
    )
    return sum((count + batch_size - 1) // batch_size for count in lengths.values())


def verified_result(
    work, fold, mix, family, plan_sha, expected_steps, expected_rows, train_sha, recipe
):
    folder = work / fold / mix / family
    result_path = folder / "validation.json"
    if not result_path.exists():
        return None
    result = read_json(result_path)
    model = read_json(folder / "model.json")
    if (
        result["model"] != model
        or model["plan_sha256"] != plan_sha
        or model["fold"] != fold
        or model["positive_mix"] != mix
        or model["train_ids_sha256"] != train_sha
        or file_hash(folder / "head.npz") != model["parameters_sha256"]
    ):
        raise ValueError("Published model, score receipt or classifier identity changed")
    if family == "fine-tuned":
        epochs = recipe["epochs"]
        training = read_json(folder / "training-complete.json")
        frozen = read_json(work / fold / mix / "frozen/model.json")
        if (
            training != result["training"]
            or training["identity"]["recipe"] != recipe
            or training["identity"]["base_encoder"] != frozen["encoder"]
            or training["identity"]["plan_sha256"] != plan_sha
            or training["identity"]["train_ids_sha256"] != train_sha
            or len(training["history"]) != epochs
            or [r["epoch"] for r in training["history"]] != list(range(1, epochs + 1))
            or training["optimizer_steps"] != expected_steps
            or training["training_rows"] != expected_rows
            or not 1 <= training["selected_epoch"] <= epochs
            or model["encoder"]["selected_epoch"] != training["selected_epoch"]
            or not training["all_selected_crops_visited_each_epoch"]
            or training["initial_encoder_state_digest"] == training["encoder_state_digest"]
            or training["trainable_encoder_parameters"] != 90311792
            or file_hash(folder / "best.pt") != model["selected_encoder_sha256"]
            or model["selected_encoder_sha256"] != training["best_parameters_sha256"]
        ):
            raise ValueError("A complete full-backbone three-epoch result has not been established")
    return result


def metric_row(fold, mix, family, result, training):
    field = result["validation"]["by_source"][fold]
    fsd = result["validation"]["by_source"]["fsd50k-eval"]
    return dict(
        encoder=family,
        held_source=fold,
        positive_mix=mix,
        positive_training_clips=sum(
            n for k, n in training["by_source_label"].items() if k.endswith("/drone")
        ),
        positive_training_groups=training["positive_groups"],
        FSD_FP=fsd["confusion"]["fp"],
        FSD_N=fsd["negative_clips"],
        field_TP=field["confusion"]["tp"],
        field_P=field["positive_clips"],
        field_FP=field["confusion"]["fp"],
        field_N=field["negative_clips"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=Path("work/beats-adaptation-v1"))
    p.add_argument("--output", type=Path, default=Path("reports/beats-adaptation-v1"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = read_json(args.work / "plan.json")
    plan_sha = file_hash(args.work / "plan.json")
    rows = read_jsonl(args.work / "manifest.jsonl")
    if cohort_signature(rows) != plan["recipe"]["cohort_signature"]:
        raise ValueError("Report cohort differs from the declared study")
    recipe = plan["recipe"]["fine_tuning"]
    frozen = read_json(args.work / "frozen-controls.json")
    values, completed = [], {}
    total_batches, total_visits = 0, 0
    for fold in FOLDS:
        completed[fold] = {}
        for mix in MIXES:
            ids = plan["folds"][fold][mix + "_positive_train"]
            training = partition_summary(rows, ids)
            steps = batches(rows, ids, recipe["batch_size"]) * recipe["epochs"]
            total_batches += steps
            total_visits += len(ids) * recipe["epochs"]
            for family in ["frozen", "fine-tuned"]:
                result = verified_result(
                    args.work, fold, mix, family, plan_sha, steps, len(ids), digest(ids), recipe
                )
                if family == "frozen" and (result is None or result != frozen[fold][mix]):
                    raise ValueError("The paired frozen control is missing or inconsistent")
                if result is None:
                    continue
                values.append(metric_row(fold, mix, family, result, training))
                if family == "fine-tuned":
                    completed[fold][mix] = result
    count = sum(len(v) for v in completed.values())
    state = execution(args.work, total_batches)
    state["completed_fine_tuned_evaluations"] = count
    state["planned_fine_tuned_evaluations"] = 4
    if count == 4:
        state["status"] = "complete"
    report = dict(
        captured_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        plan_sha256=plan_sha,
        corpus_rows=len(rows),
        training_crop_visits=total_visits,
        training_batches=total_batches,
        scope="Consumed-corpus development diagnostics with unchanged weak publisher labels",
        execution=state,
        complete=count == 4,
        completed_fine_tuned_controls=completed,
    )
    write_json(args.output / "comparison.json", report)
    lines = [
        "# BEATs adaptation: completed controls and execution status",
        "",
        f"Captured at {datetime.datetime.fromisoformat(report['captured_at_utc']).astimezone(ZoneInfo('America/New_York')).isoformat()} "
        f"(America/New_York). Execution status: **{state['status']}**. "
        f"Completed fine-tuned evaluations: **{count}/4**.",
        "",
        f"The full cohort has **{len(rows):,} crops**. The declared neural study has "
        f"**{total_visits:,} training-crop visits across {total_batches:,} batches** and "
        f"{recipe['epochs']} complete epochs per arm. An unfinished arm contributes no evaluation row.",
        "",
        "These are known-corpus development comparisons, not fresh generalization tests. "
        "Each fold excludes its whole held field source from fitting and model selection. "
        "Negative training and selection are matched within each fold.",
        "",
        "| Encoder | Held source | Positive mix | Positive crops / groups | FSD crop FPR | Field drone recall | Field background FPR |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for value in values:
        lines.append(
            f"| {value['encoder']} | {value['held_source']} | {value['positive_mix']} | "
            f"{value['positive_training_clips']:,} / {value['positive_training_groups']} | "
            f"{pct(value['FSD_FP'] / value['FSD_N'])} | "
            f"{pct(value['field_TP'] / value['field_P']) if value['field_P'] else '—'} | "
            f"{pct(value['field_FP'] / value['field_N']) if value['field_N'] else '—'} |"
        )
    if (args.output / "MATCHED-RUNTIME.md").exists():
        lines += ["", "[Runtime-control provenance](MATCHED-RUNTIME.md) distinguishes these matched "
                  "native-runtime controls from the earlier cached-CPU references."]
    for fold in FOLDS:
        for mix in completed[fold]:
            paired = Path("paired-analysis") / fold / mix / "README.md"
            if (args.output / paired).exists():
                lines += ["", f"[Complete paired error analysis: {fold}/{mix}]({paired.as_posix()})."]
    lines += [
        "",
        "The frozen controls show that adding drone-positive sources does not consistently improve transfer. "
        "Completed adapted rows, when available, must be compared with their matching frozen row at the same selection rule. "
        "Incomplete training, synthetic software checks and training-batch loss are not detector performance results.",
        "",
        "## Execution and compute",
        "",
    ]
    if "accumulated_hours" in state:
        lines.append(
            f"The supervisor has accumulated {state['accumulated_hours']:.2f} hours against a "
            f"{state['max_wall_hours']:g}-hour cumulative budget. Its active child is "
            f"{'verified' if state['active_child_verified'] else 'not present'}. "
            f"Recovery checkpoints are configured every {state['checkpoint_every_steps']:,} steps. "
            "A saved 'running' status is treated as interrupted when its matching process is absent."
        )
    timing = state.get("first_training_segment_timing")
    if timing:
        lines += [
            "",
            f"Actual first-arm training logged {timing['observed_steps']:,} steps in "
            f"{timing['elapsed_seconds'] / 3600:.2f} hours "
            f"({timing['seconds_per_batch']:.2f} seconds/batch). Extrapolating to all four arms gives "
            f"about **{timing['projected_training_only_hours']:.0f} hours of training alone**, "
            "before selection, feature extraction or interruption replay. This supersedes the "
            "initial 30-hour estimate from repeated synthetic batches. At the observed rate the "
            "full CPU study will not finish inside its 44-hour cap; the cap remains in force.",
        ]
    if state.get("last_resume_checkpoint_steps") is not None:
        lines += [
            "",
            f"The most recent recovery logged an exact resume from optimizer step "
            f"{state['last_resume_checkpoint_steps']:,}. Unsaved batches after a checkpoint are "
            "replayed; they do not become extra training epochs or independent evidence.",
        ]
    if state.get("last_logged_batch") is not None:
        last = state["last_logged_batch"]
        lines += [
            "",
            f"The latest logged training position is epoch {last['epoch']}, "
            f"batch {last['batch']:,}/{last['batches']:,}, optimizer step {last['steps']:,}.",
        ]
    lines += [
        "",
        "The native frontend, source folds, three-epoch recipe and calibration rule are unchanged. "
        "The best trained epoch is selected on weighted selection-DEV log-loss; a C1 head is then "
        "refitted on TRAIN embeddings. Held-field/FSD scores never select that epoch or cutoff.",
        "",
        "Positive training spans four, five or seven capture groups. The same 7,931 FSD recordings "
        "appear in both folds; they are not 15,862 independent negatives. WSU background windows "
        "come from one recording. Publisher presence labels have not received independent "
        "crop-audibility review. Crop rates do not establish false alarms/hour or event recall.",
        "",
    ]
    (args.output / "RESULTS.md").write_text("\n".join(lines))
    for name, subset, exclude in [
        ("metrics.csv", values, set()),
        ("frozen-metrics.csv", [v for v in values if v["encoder"] == "frozen"], {"encoder"}),
    ]:
        fields = [key for key in subset[0] if key not in exclude]
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in subset)
    write_json(args.output / "frozen-controls.json", frozen)
    for fold in FOLDS:
        for mix in MIXES:
            source = args.work / fold / mix / "frozen"
            destination = args.output / "heads" / fold / mix
            destination.mkdir(parents=True, exist_ok=True)
            for name in ["head.npz", "model.json"]:
                shutil.copyfile(source / name, destination / name)
            if mix in completed[fold]:
                source = args.work / fold / mix / "fine-tuned"
                destination = args.output / "heads-fine-tuned" / fold / mix
                destination.mkdir(parents=True, exist_ok=True)
                for name in ["head.npz", "model.json", "training-complete.json"]:
                    shutil.copyfile(source / name, destination / name)
    print(args.output / "RESULTS.md")


if __name__ == "__main__":
    main()
