"""Controlled, explicitly synthetic experiments and a frozen comparison."""

import time
from pathlib import Path

import numpy as np

from .common import file_hash, hardware, read_json, revision, write_json
from .data import audit, checked_audio, load_manifest, split_manifest
from .demo import build_demo
from .evaluation import evaluate, tune
from .model import train
from .synthetic import generate_corpus


def gain_probe(model, manifest):
    records = [r for r in load_manifest(manifest, True) if r["split"] == "development"]
    deltas, flips, count = [], 0, 0
    gains_db = [-24, -12, 6]
    for row in records:
        audio = checked_audio(row, manifest, model.config.sample_rate)
        # Fixed center-context probe, selected without consulting outcomes.
        middle = max(0, len(audio) // 2 - model.config.window_samples // 2)
        chunk = audio[middle:middle + model.config.window_samples]
        score = model.score(chunk)
        for db in gains_db:
            modified = chunk * 10 ** (db / 20)
            if np.max(np.abs(modified)) > .98:
                continue  # No clipping; isolate scalar gain only.
            changed = model.score(modified)
            deltas.append(abs(changed - score))
            threshold = model.metadata["decision"]["on_threshold"]
            flips += (score >= threshold) != (changed >= threshold)
            count += 1
    return {"split": "development", "gains_db": gains_db, "comparisons": count,
            "mean_absolute_score_change": float(np.mean(deltas)) if deltas else None,
            "max_absolute_score_change": float(np.max(deltas)) if deltas else None,
            "threshold_flip_fraction": flips / count if count else None,
            "scope": "Fixed central 3 s context; representation sensitivity only, no full-stream or authentic robustness claim"}


def run_synthetic(output, contract, sessions_per_source=10):
    started = time.perf_counter()
    output = Path(output)
    if output.exists():
        raise ValueError("Use a fresh experiment output directory")
    output.mkdir(parents=True)
    plan = {
        "schema_version": 1, "purpose": "SYNTHETIC SOFTWARE/CONTROLLED-SIGNAL CHECKS ONLY",
        "code": revision(), "contract_file_sha256": file_hash(contract),
        "generation_seed": 20261108, "split_seed": 42,
        "reserved_final_source": "synthetic-generator-2",
        "arms": [
            {"name": "absolute", "representation": "absolute_mel", "seed": 42,
             "augment": False, "hits": [1], "hypothesis": "Inexpensive absolute log-Mel baseline"},
            {"name": "relative", "representation": "relative_mel", "seed": 42,
             "augment": False, "hits": [1], "hypothesis": "Subtracting scalar log-energy removes gain shortcut"},
            {"name": "harmonic", "representation": "relative_harmonic", "seed": 42,
             "augment": False, "hits": [1], "hypothesis": "Flatness/crest/cepstral features distinguish tonal structure"},
            {"name": "temporal", "representation": "relative_mel", "seed": 42,
             "augment": False, "hits": [1, 2], "hypothesis": "Confirmation reduces transient alarms at a delay cost"},
            *[{"name": f"gain-aug-{seed}", "representation": "relative_mel", "seed": seed,
               "augment": True, "hits": [1, 2],
               "hypothesis": "Train-only gain augmentation; repeated seeds check stochastic sensitivity"}
              for seed in [7, 42, 123]],
        ],
        "selection": "Compare seed-42 arms only; prefer point FA/h <= contract limit, highest development event recall, lowest FA/h, fewest gain threshold flips, lowest gain score change, lowest delay, then arm name. Other augmentation seeds are sensitivity checks, not cherry-picked candidates.",
        "final": "One frozen batch: baseline and development-selected candidate. No tuning on final.",
        "notes": "Unaugmented lbfgs is deterministic; repeated nominal seeds would not be independent experiments. No expensive model expansion on generated signals.",
    }
    write_json(output / "experiment-plan.json", plan)
    annotations = generate_corpus(output / "corpus", sessions_per_source)
    audit(output / "corpus", output / "corpus" / "inventory-v1.jsonl", annotations)
    manifest = output / "corpus" / "manifest-v1.jsonl"
    split_manifest(output / "corpus" / "inventory-v1.jsonl", manifest, final_sources=["synthetic-generator-2"])
    rows = load_manifest(manifest, True)
    final_sessions = sorted({r["session_id"] for r in rows if r["split"] == "final"})[:2]
    fixed = {"recording_ids": sorted(r["id"] for r in rows if r["split"] == "final"
                                    and r["session_id"] in final_sessions),
             "reason": "First two final generator sessions by ID, all three recordings per session; selected before training and final inference. Positive, ordinary-negative and difficult-negative generated inputs.",
             "manifest_sha256": file_hash(manifest)}
    write_json(output / "fixed-demo.json", fixed)
    ledger = []
    frozen_paths = {}
    for arm in plan["arms"]:
        begin = time.perf_counter()
        raw_path = output / "models" / (arm["name"] + "-trained")
        frozen = output / "models" / arm["name"]
        train(manifest, raw_path, arm["representation"], arm["seed"], arm["augment"])
        model = tune(raw_path, manifest, contract, frozen, arm["hits"])
        frozen_paths[arm["name"]] = frozen
        result = {
            **arm, "code": plan["code"], "manifest_sha256": file_hash(manifest),
            "model_id": model.model_id, "features": model.metadata["features"],
            "decision": model.metadata["decision"],
            "development": model.metadata["selection"]["chosen"]["metrics"],
            "gain_probe": gain_probe(model, manifest),
            "compute_seconds": time.perf_counter() - begin,
            "training_window_counts": model.metadata["training"]["window_counts"],
            "hardware": hardware(),
        }
        ledger.append(result)
        write_json(output / "experiment-ledger.json", ledger)
        print(f"{arm['name']}: development recall={result['development']['event_recall']:.3f}, "
              f"FA/h={result['development']['false_alarms_per_hour']:.2f}, "
              f"gain flips={result['gain_probe']['threshold_flip_fraction']:.3f}", flush=True)

    def rank(result):
        metric, probe = result["development"], result["gain_probe"]
        rate, recall = metric["false_alarms_per_hour"], metric["event_recall"]
        delay = metric["p95_delay_seconds_detected_events_only"]
        feasible = rate <= read_json(contract)["targets"]["false_alarms_per_hour_upper_95"]
        return (not feasible, -recall if feasible else rate, rate if feasible else -recall,
                probe["threshold_flip_fraction"], probe["mean_absolute_score_change"],
                delay if delay is not None else 1e12, result["name"])

    winner = min([item for item in ledger if item["seed"] == 42], key=rank)
    comparison_names = list(dict.fromkeys(["absolute", winner["name"]]))
    selection = {"selected": winner["name"], "baseline": "absolute", "rule": plan["selection"],
                 "models": {name: next(r["model_id"] for r in ledger if r["name"] == name)
                            for name in comparison_names}, "final_touched": False}
    write_json(output / "frozen-selection.json", selection)
    final = evaluate([frozen_paths[name] for name in comparison_names], manifest, "final", output / "final")
    chosen_final = next(r for r in final["results"] if r["model_id"] == winner["model_id"])
    failures = sorted(r["id"] for r in chosen_final["records"] if r["recording_error"])
    demo_selection = {**fixed, "recording_ids": list(dict.fromkeys(fixed["recording_ids"] + failures)),
                      "posthoc_failure_ids": [r for r in failures if r not in fixed["recording_ids"]],
                      "reason": fixed["reason"] + " Post-hoc appendix includes every candidate holdout failure, not additional favorable examples."}
    write_json(output / "demo-selection.json", demo_selection)
    build_demo(frozen_paths[winner["name"]], manifest, output / "demo-selection.json", output / "demo.html")
    report = {
        "schema_version": 1, "scope": plan["purpose"], "selected": winner["name"],
        "manifest_sha256": file_hash(manifest), "development": ledger, "final": final,
        "seconds": time.perf_counter() - started, "hardware": hardware(),
        "claim": "No measured reliability for authentic drones. No real microphones, distance, outdoor environment or SNR range validated.",
        "missing": ["Original WAV archive with provenance/permissions", "Verified temporal annotations",
                    "Independent authentic positive events and long hard-negative recordings",
                    "Accessible microphone hardware for actual capture"],
    }
    write_json(output / "results.json", report)
    write_report(output / "RESULTS.md", report)
    return report


def write_report(output, report):
    lines = ["# Controlled synthetic results", "", report["claim"], "",
             f"Selected on development only: **{report['selected']}**. One final batch compared it with the frozen baseline.",
             "", "## Development ablations", "",
             "| Arm | Recall | FA/h | p95 delay (s) | Gain score drift | Gain flips | Runtime (s) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm in report["development"]:
        m, p = arm["development"], arm["gain_probe"]
        delay = m["p95_delay_seconds_detected_events_only"]
        lines.append(f"| {arm['name']} | {m['event_recall']:.3f} | {m['false_alarms_per_hour']:.2f} | "
                     f"{delay if delay is not None else 'n/a'} | {p['mean_absolute_score_change']:.6f} | "
                     f"{p['threshold_flip_fraction']:.3f} | {arm['compute_seconds']:.2f} |")
    lines += ["", "## Frozen generator holdout", "",
              "This is a held-out signal generator, not an independent acoustic dataset. Thresholds and alarm confirmation were frozen before scoring.", ""]
    for arm in report["final"]["results"]:
        m, uncertainty = arm["summary"], arm["summary"]["uncertainty"]
        name = next(item["name"] for item in report["development"] if item["model_id"] == arm["model_id"])
        lines += [f"### {name}", "", f"Model `{arm['model_id']}`.", "",
                  f"- Decision: `{arm['decision']}`.",
                  f"- {m['recordings']} files / {m['independent_groups']} generator session groups; {m['events']} eligible events, {m['negative_hours']:.4f} negative hours.",
                  f"- Recall {m['event_recall']:.3f} ({m['matched_events']}/{m['events']}); cluster 95% interval {uncertainty['recall_cluster_95']}; conservative lower bound {uncertainty['recall_conservative_lower_95']:.3f}.",
                  f"- {m['false_alarm_events']} false alarm events; {m['false_alarms_per_hour']:.2f}/h; conservative upper 95% bound {uncertainty['fa_per_hour_conservative_upper_95']:.2f}/h.",
                  f"- p95 delay among detections {m['p95_delay_seconds_detected_events_only']} s; {m['recording_errors']} recording-level errors; precision {m['event_precision']}.",
                  f"- End-to-end real-time factor {arm['runtime']['real_time_factor']:.4f}; hardware {arm['runtime']['hardware']['cpu']}.",
                  f"- Numerical checks: `{arm['assessment']['numerical_checks']}`. Evidence checks: `{arm['assessment']['evidence_requirements']}`.", "", "Observed failures:", ""]
        failed = [record for record in arm["records"] if record["recording_error"]]
        if not failed:
            lines += ["No errors on this small synthetic holdout; this supplies no authentic capture evidence."]
        for record in failed[:12]:
            lines.append(f"- `{record['id']}` ({record['source']}): {len(record['missed_event_indices'])} missed events, {record['false_alarm_count']} false alarms; alarm intervals {record['alarms']}.")
        lines.append("")
    lines += ["## Interpretation and next experiment", "",
              "Per-window relative log-Mel removes scalar gain from the representation by construction; the gain probe measures that software property. Tonal/harmonic features are also activated by generated motor/fan-like negatives. These observations do not establish the cause of historical failures or an advantage on authentic capture.", "",
              "The current experiment compares representations and alarm policies on fixed development recordings. Seed variants only apply to stochastic gain augmentation. Larger audio-pretrained models are deferred: there is no verified audio with which to assess generalization. No distances, SNR range, microphone coverage, drone coverage or field-readiness claim is supported.", "",
              "Next: audit the original audio archive and recover original/session identities; independently annotate full recordings, then rerun the frozen protocol on authentic audio with 100+ events, 30+ groups and 10+ negative hours across at least two sources/devices and three drone types. See docs/collection-protocol.md. Do not relabel generated signals as collected drone data.", "",
              "The demo's six fixed recordings were chosen by ID before training; all positives and ordinary/difficult negatives from those sessions are included. A separately labeled post-hoc appendix includes every candidate holdout failure. All benchmark traces are retained. The demo is a convenience, not the benchmark.", ""]
    Path(output).write_text("\n".join(lines))
