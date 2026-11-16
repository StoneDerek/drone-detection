#!/usr/bin/env python3
"""Run the complete, predeclared BEATs adaptation arms on an explicit runtime."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from drone_detector.adaptation import cohort_signature
from drone_detector.beats_encoder import BEATsEncoder
from drone_detector.beats_training import (
    TaskModel,
    build_model,
    configure,
    embed_selected,
    read_state,
    matrix_metadata,
    save_matrix,
    run_identity,
    train,
)
from drone_detector.common import digest, file_hash, read_json, write_json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_beats_adaptation import fit_head, load_rows
from run_confuser_study import broad_cutoff
from run_corpus_study import logits, summaries


def base_matrix(args):
    rows, saved = load_rows(args)
    out = args.work / "base-embeddings.npz"
    loader = BEATsEncoder(args.assets, args.device)
    identity = dict(loader.identity, torch_intraop_threads=args.threads)
    if out.exists():
        meta = matrix_metadata(out)
        if meta["encoder"] != identity or meta["sha256"] != file_hash(out):
            raise ValueError(
                "Existing base matrix uses a different runtime; use a separate study directory"
            )
        if meta["cohort_signature"] != cohort_signature(rows):
            raise ValueError("Base matrix belongs to a different cohort")
        return
    empty_head = dict(
        mean=np.zeros(768), scale=np.ones(768), coef=np.zeros(768), intercept=np.zeros(1)
    )
    model = TaskModel(loader._load(), empty_head).to(args.device).eval().requires_grad_(False)
    x = embed_selected(
        model,
        rows,
        args.device,
        saved["recipe"]["fine_tuning"]["batch_size"],
        stop_path=args.work / "pause-requested.json",
    )
    save_matrix(
        out,
        x,
        [r["id"] for r in rows],
        dict(
            cohort_signature=cohort_signature(rows),
            encoder=identity,
            rows=len(rows),
            dimensions=768,
            extraction="released native frontend; exact-length batches; no waveform padding",
            implementation_sha256=file_hash(__file__),
        ),
    )


def run_arm(args, fold, mix):
    rows, saved = load_rows(args)
    spec = saved["folds"][fold]
    recipe = saved["recipe"]["fine_tuning"]
    lookup = {r["id"]: i for i, r in enumerate(rows)}
    train_ids = spec[mix + "_positive_train"]
    indices = {
        name: [lookup[i] for i in ids]
        for name, ids in [
            ("train", train_ids),
            ("selection", spec["selection"]),
            ("validation", spec["validation"]),
        ]
    }
    frozen = args.work / fold / mix / "frozen"
    initial = read_json(frozen / "model.json")
    if file_hash(frozen / "head.npz") != initial["parameters_sha256"]:
        raise ValueError("Paired initial classification head changed")
    output = args.work / fold / mix / "fine-tuned"
    output.mkdir(parents=True, exist_ok=True)
    configure(args.device, recipe["seed"], args.threads)
    model, base = build_model(args.assets, frozen / "head.npz", args.device)
    base["torch_intraop_threads"] = args.threads
    if base != initial["encoder"]:
        raise ValueError("Frozen and adapted arms must use the same native numerical runtime")
    identity = run_identity(
        base,
        recipe,
        fold,
        mix,
        file_hash(args.work / "plan.json"),
        args.device,
        args.threads,
        train_ids,
    )
    identity["checkpoint_every_steps"] = args.checkpoint_every
    identity["driver_sha256"] = file_hash(__file__)
    if args.device == "cuda":
        identity["cuda_device_name"] = torch.cuda.get_device_name(0)
        identity["cuda_device_capability"] = list(torch.cuda.get_device_capability(0))
    completion = train(
        model,
        [rows[i] for i in indices["train"]],
        [rows[i] for i in indices["selection"]],
        recipe,
        output,
        identity,
        args.device,
        checkpoint_every=args.checkpoint_every,
        stop_path=args.work / "pause-requested.json",
    )
    if completion.get("paused_for_budget_or_user_request"):
        write_json(output / "paused.json", completion)
        print("Training paused at a verified batch checkpoint", json.dumps(completion), flush=True)
        return dict(status="paused", **completion)
    selected = read_state(output / "best.pt")
    model.encoder.load_state_dict(selected["encoder"], strict=True)
    model.encoder.eval().requires_grad_(False)
    matrix_path = output / "embeddings.npz"
    weights_sha = file_hash(output / "best.pt")
    if matrix_path.exists():
        meta = matrix_metadata(matrix_path)
        if (
            meta["sha256"] != file_hash(matrix_path)
            or meta["adapted_parameters_sha256"] != weights_sha
            or meta["cohort_signature"] != cohort_signature(rows)
        ):
            raise ValueError("Adapted feature identity changed")
        with np.load(matrix_path, allow_pickle=False) as z:
            if z["ids"].tolist() != [r["id"] for r in rows]:
                raise ValueError("Adapted feature order changed")
            x = z["x"].astype(np.float64)
    else:
        x = embed_selected(
            model,
            rows,
            args.device,
            recipe["batch_size"],
            stop_path=args.work / "pause-requested.json",
        )
        save_matrix(
            matrix_path,
            x,
            [r["id"] for r in rows],
            dict(
                cohort_signature=cohort_signature(rows),
                adapted_parameters_sha256=weights_sha,
            ),
        )
        x = x.astype(np.float64)
    head = fit_head(x[indices["train"]], [rows[i] for i in indices["train"]])
    selection_scores = logits(x[indices["selection"]], head)
    cutoff, components = broad_cutoff(selection_scores, [rows[i] for i in indices["selection"]])
    np.savez_compressed(output / "head.npz", **head)
    encoder = dict(
        base,
        family="beats-task-adapted",
        adapted_parameters_sha256=weights_sha,
        selected_epoch=completion["selected_epoch"],
        adaptation_identity_sha256=digest(identity),
    )
    meta = dict(
        fold=fold,
        positive_mix=mix,
        encoder=encoder,
        parameters_sha256=file_hash(output / "head.npz"),
        cutoff_logit=cutoff,
        component_cutoffs=components,
        plan_sha256=file_hash(args.work / "plan.json"),
        train_ids_sha256=digest(train_ids),
        selected_encoder_sha256=weights_sha,
        scope="known-source development validation; held source excluded from fitting and selection",
    )
    # The final head and cutoff are fixed before scores on the held source.
    write_json(output / "model.json", meta)
    validation_scores = logits(x[indices["validation"]], head)
    result = dict(
        model=meta,
        training=completion,
        selection=summaries([rows[i] for i in indices["selection"]], selection_scores, cutoff),
        validation=summaries([rows[i] for i in indices["validation"]], validation_scores, cutoff),
    )
    write_json(output / "validation.json", result)
    print(
        "Completed full adaptation arm",
        fold,
        mix,
        json.dumps(result["validation"]["by_source"]),
        flush=True,
    )
    if not args.keep_completed_optimizer:
        # Completed inference artifacts remain. These are this run's temporary
        # optimizer recovery files, retired only after successful final scoring.
        for name in ["latest.pt", "latest.pt.json", "latest.pt.pending.json"]:
            (output / name).unlink(missing_ok=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["base", "fit", "all"])
    p.add_argument("--work", type=Path, default=Path("work/beats-adaptation-v1"))
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--fold", choices=["wsu-air3", "audrok-dresden"])
    p.add_argument("--positive-mix", choices=["current", "broader"])
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--keep-completed-optimizer", action="store_true")
    args = p.parse_args()
    configure(args.device, threads=args.threads)
    if args.stage == "base":
        base_matrix(args)
    elif args.stage == "fit":
        if args.fold is None or args.positive_mix is None:
            p.error("fit requires a held source and positive mix")
        result = run_arm(args, args.fold, args.positive_mix)
        if result.get("status") == "paused":
            raise SystemExit(75)
    else:
        results = {}
        for fold in ["audrok-dresden", "wsu-air3"]:
            results[fold] = {}
            for mix in ["current", "broader"]:
                results[fold][mix] = run_arm(args, fold, mix)
                if results[fold][mix].get("status") == "paused":
                    raise SystemExit(75)
        write_json(args.work / "fine-tuned-controls.json", results)


if __name__ == "__main__":
    main()
