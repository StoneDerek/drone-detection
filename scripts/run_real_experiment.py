#!/usr/bin/env python3
"""Real-audio training/development and one prespecified frozen assessment batch."""

import argparse
import time
from pathlib import Path

from threadpoolctl import threadpool_limits

from drone_detector.common import file_hash, read_json, revision, write_json
from drone_detector.data import load_manifest
from drone_detector.model import load_model
from drone_detector.weak import evaluate_weak, train_weak, tune_weak

ARMS = {"absolute": "absolute_mel", "relative": "relative_mel",
        "harmonic": "relative_harmonic", "yamnet": "yamnet"}


def rank(name, model):
    m = model.metadata["selection"]["chosen"]["metrics"]
    fpr, recall = m["negative_recording_false_positive_fraction"], m["recording_recall"]
    preference = {"yamnet": 0, "relative": 1, "harmonic": 2, "absolute": 3}
    return (fpr > .05, -recall if fpr <= .05 else fpr, fpr if fpr <= .05 else -recall,
            -m["roc_auc_max_window_score"], preference[name])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="runs/real-v1/corpus/manifest-v1.jsonl")
    p.add_argument("--workdir", default="runs/real-v1")
    p.add_argument("--development-only", action="store_true")
    p.add_argument("--reuse-trained", action="store_true")
    args = p.parse_args()
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    if not args.development_only and Path(args.manifest).with_suffix(".weak-final-use.json").exists():
        raise ValueError("Final already consumed; use --development-only for further experiments")
    rows = load_manifest(args.manifest, True)
    models, ledger = {}, []
    with threadpool_limits(limits=1):
        for name, representation in ARMS.items():
            begin = time.perf_counter()
            trained, frozen = work / "models" / (name + "-trained"), work / "models" / name
            if args.reuse_trained and frozen.exists():
                model = load_model(frozen, True)
                if model.metadata["training"]["manifest_sha256"] != file_hash(args.manifest):
                    raise ValueError("Existing model uses a different corpus")
            else:
                encoder = work / "encoders/yamnet" if name == "yamnet" else None
                train_weak(args.manifest, trained, representation, encoder_source=encoder)
                model = tune_weak(trained, args.manifest, frozen)
            models[name] = model
            ledger.append({"arm": name, "model_id": model.model_id,
                "training": model.metadata["training"], "code": model.metadata["code"],
                "representation": representation, "decision": model.metadata["decision"],
                "development": model.metadata["selection"]["chosen"]["metrics"],
                "training_seconds": model.metadata["training"]["seconds"],
                "development_seconds": model.metadata["selection"]["runtime_seconds"],
                "orchestration_seconds": time.perf_counter() - begin})
        selected = min(models, key=lambda name: rank(name, models[name]))
        references = list(dict.fromkeys(["absolute", selected, "yamnet"]))
        frozen_selection = {"selected": selected, "models": {name: models[name].model_id for name in references},
            "manifest_sha256": file_hash(args.manifest), "code": revision(),
            "selection_rule": read_json(work / "experiment-plan.json")["selection"],
            "final_touched_at_selection": False}
        write_json(work / "experiment-ledger.json", ledger)
        if (work / "frozen-selection.json").exists():
            raise ValueError("Refusing to overwrite a frozen selection")
        write_json(work / "frozen-selection.json", frozen_selection)
        final_rows = [r for r in rows if r["split"] == "final"]
        demo = []
        for source, label, kind in [
            ("svanstrom", "drone", None), ("svanstrom", "non_drone", "background"),
            ("svanstrom", "non_drone", "helicopter"), ("batear-field", "drone", None),
            ("batear-field", "non_drone", None), ("esc10", "non_drone", "chain_saw"),
            ("d3set-ground", "drone", None), ("d3set-noise", "non_drone", None),
        ]:
            match = next((r for r in final_rows if r["source"] == source and r["label"] == label
                          and (kind is None or r.get("negative_kind") == kind)
                          and r["duration_seconds"] <= 600), None)
            if match:
                demo.append(match["id"])
        write_json(work / "fixed-demo.json", {"recording_ids": demo,
            "manifest_sha256": file_hash(args.manifest),
            "reason": "First metadata-matching inputs per prespecified source/class/negative type, chosen before final scoring. Publisher labels remain weak."})
        if args.development_only:
            print(f"Development complete; selected {selected}. Final not assessed.")
            return
        result = evaluate_weak([work / "models" / name for name in references], args.manifest,
                               "final", work / "final")
        result["selected"] = selected
        write_json(work / "results.json", result)
        for name, entry in zip(references, result["results"], strict=True):
            print(name, entry["summary"], flush=True)


if __name__ == "__main__":
    main()
