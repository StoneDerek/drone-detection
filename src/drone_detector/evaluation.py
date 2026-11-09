"""Continuous-recording metrics, development-only selection and a final-use ledger."""

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta, chi2

from .common import (
    digest, file_hash, hardware, overlap, read_json, read_jsonl,
    revision, union_intervals, write_json, write_jsonl,
)
from .data import VERIFIED, checked_audio, load_manifest
from .model import load_model, save_model
from .stream import Alarm, alarm_intervals, replay_audio


def apply_decision(rows, decision):
    alarm = Alarm(decision)
    return [{**row, "alarm": alarm.step(row["score"])} for row in rows]


def record_metrics(record, trace, contract):
    duration = record["duration_seconds"]
    collar = contract["annotation"]["onset_tolerance_seconds"]
    minimum = contract["annotation"]["minimum_claimed_event_seconds"]
    events = [event for event in record["events"] if event[1] - event[0] >= minimum]
    alarms = alarm_intervals(trace, duration, contract["decision"]["merge_gap_seconds"])
    excluded = union_intervals(
        [[a - collar, b + collar] for a, b in record["events"]] + record["ambiguous"], duration
    )
    negative_seconds = duration - sum(b - a for a, b in excluded)
    used, matches, delays = set(), [], []
    for index, (start, end) in enumerate(events):
        found = next((i for i, (a, _) in enumerate(alarms)
                      if i not in used and start - collar <= a <= end + collar
                      and not any(c <= a < d for c, d in record["ambiguous"])), None)
        if found is not None:
            used.add(found)
            onset = alarms[found][0]
            trigger = next(row for row in trace if row["alarm"] and
                           row["timestamp_seconds"] == onset)
            # Includes window/context, scheduling on the 0.5 s grid, block
            # availability, and measured feature/scoring/decision compute.
            delay = max(0.0, trigger["delivered_at_seconds"] - start)
            delays.append(delay)
            matches.append({"event_index": index, "alarm_index": found, "delay_seconds": delay})
    false_alarms = [i for i, (a, _) in enumerate(alarms)
                    if i not in used and not any(c <= a <= d for c, d in excluded)]
    false_time = sum(max(0.0, b - a - overlap(a, b, excluded)) for a, b in alarms)
    eligible_alarm_count = sum(not any(c <= a < d for c, d in record["ambiguous"])
                               for a, _ in alarms)
    return {
        "id": record["id"], "source": record["source"], "group_id": record["group_id"],
        "origin": record["origin"], "duration_seconds": duration,
        "negative_seconds": negative_seconds,
        "event_count": len(events), "all_event_count": len(record["events"]),
        "short_event_count": len(record["events"]) - len(events),
        "matched_events": len(matches), "false_alarm_count": len(false_alarms),
        "alarm_count": eligible_alarm_count,
        "negative_alarm_seconds": false_time, "delays_seconds": delays,
        "alarms": alarms, "events": events, "matches": matches,
        "missed_event_indices": [i for i in range(len(events))
                                 if i not in {m["event_index"] for m in matches}],
        "false_alarm_indices": false_alarms,
        "recording_error": len(matches) != len(events) or bool(false_alarms),
        "recording_binary_truth": bool(record["events"]),
        "recording_binary_prediction": bool(eligible_alarm_count),
    }


def summarize(records, uncertainty=True):
    if not records:
        raise ValueError("Zero evaluated recordings is not a successful evaluation")
    count = sum(r["event_count"] for r in records)
    matched = sum(r["matched_events"] for r in records)
    fa = sum(r["false_alarm_count"] for r in records)
    negative = sum(r["negative_seconds"] for r in records)
    delays = [delay for r in records for delay in r["delays_seconds"]]
    alarms = sum(r["alarm_count"] for r in records)
    tp = sum(r["recording_binary_truth"] and r["recording_binary_prediction"] for r in records)
    fp = sum(not r["recording_binary_truth"] and r["recording_binary_prediction"] for r in records)
    fn = sum(r["recording_binary_truth"] and not r["recording_binary_prediction"] for r in records)
    result = {
        "recordings": len(records), "independent_groups": len({r["group_id"] for r in records}),
        "hours": sum(r["duration_seconds"] for r in records) / 3600,
        "negative_hours": negative / 3600,
        "events": count, "all_events": sum(r["all_event_count"] for r in records),
        "short_events_outside_claim": sum(r["short_event_count"] for r in records),
        "matched_events": matched, "false_alarm_events": fa, "alarm_events": alarms,
        "event_recall": matched / count if count else None,
        "event_precision": matched / alarms if alarms else None,
        "false_alarms_per_hour": fa * 3600 / negative if negative > 0 else None,
        "negative_alarm_time_fraction": sum(r["negative_alarm_seconds"] for r in records) / negative
        if negative > 0 else None,
        "p95_delay_seconds_detected_events_only": float(np.quantile(delays, 0.95)) if delays else None,
        "median_delay_seconds_detected_events_only": float(np.median(delays)) if delays else None,
        "recording_errors": sum(r["recording_error"] for r in records),
        "recording_precision": tp / (tp + fp) if tp + fp else None,
        "recording_recall": tp / (tp + fn) if tp + fn else None,
        "recording_confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": len(records) - tp - fp - fn},
    }
    if not uncertainty:
        return result
    # Exposure interval is explicitly a stationary Poisson approximation, not
    # evidence that correlated recordings or future field conditions are covered.
    poisson = [float(0.5 * chi2.ppf(0.025, 2 * fa)) if fa else 0.0,
               float(0.5 * chi2.ppf(0.975, 2 * (fa + 1)))]
    poisson = [x * 3600 / negative for x in poisson] if negative > 0 else None
    binomial = [float(beta.ppf(0.025, matched, count - matched + 1)) if matched else 0.0,
                float(beta.ppf(0.975, matched + 1, count - matched)) if matched < count else 1.0]
    if not count:
        binomial = None
    grouped = defaultdict(list)
    for record in records:
        grouped[record["group_id"]].append(record)
    clusters = list(grouped.values())
    recall_samples, rate_samples, delay_samples = [], [], []
    rng = np.random.default_rng(20261108)
    for _ in range(2000):
        selected = [r for i in rng.integers(0, len(clusters), len(clusters)) for r in clusters[i]]
        n = sum(r["event_count"] for r in selected)
        exposure = sum(r["negative_seconds"] for r in selected)
        if n:
            recall_samples.append(sum(r["matched_events"] for r in selected) / n)
        if exposure > 0:
            rate_samples.append(sum(r["false_alarm_count"] for r in selected) * 3600 / exposure)
        detected = [d for r in selected for d in r["delays_seconds"]]
        if detected:
            delay_samples.append(float(np.quantile(detected, 0.95)))

    def bounds(values):
        return [float(x) for x in np.quantile(values, [0.025, 0.975])] if values else None

    recall_ci, rate_ci = bounds(recall_samples), bounds(rate_samples)
    result["uncertainty"] = {
        "method": "2000 component/session cluster bootstrap replicates, seed 20261108; intervals conditional on observed corpus",
        "recall_cluster_95": recall_ci,
        "fa_per_hour_cluster_95": rate_ci,
        "p95_delay_cluster_95": bounds(delay_samples),
        "recall_binomial_95_assumes_independent_events": binomial,
        "fa_per_hour_poisson_95_assumes_stationary_rate": poisson,
        "recall_conservative_lower_95": min(recall_ci[0], binomial[0])
        if recall_ci and binomial else None,
        "fa_per_hour_conservative_upper_95": max(rate_ci[1], poisson[1])
        if rate_ci and poisson else None,
        "warning": "Synthetic/session grouping is not proof of real-world independence. Degenerate bootstrap intervals and zero observed alarms do not prove zero risk.",
    }
    return result


def score_records(model, manifest, split):
    rows = load_manifest(manifest, require_split=True)
    if file_hash(manifest) != model.metadata["training"]["manifest_sha256"]:
        raise ValueError("Manifest differs from training; use a versioned protocol, do not silently change splits")
    selected = [r for r in rows if r["split"] == split and r["annotation_status"] in VERIFIED]
    if not selected:
        raise ValueError("No fully annotated recordings in selected split")
    traces, elapsed = {}, 0.0
    decision = {"on_threshold": 0.5, "off_threshold": 0.375, "on_hits": 1, "off_hits": 2}
    for row in selected:
        begin = time.perf_counter()
        audio = checked_audio(row, manifest, model.config.sample_rate)
        traces[row["id"]] = replay_audio(model, audio, decision)
        elapsed += time.perf_counter() - begin
    return selected, traces, elapsed


def tune(bundle, manifest, contract_path, output, on_hits_options=None):
    model, contract = load_model(bundle), read_json(contract_path)
    if model.metadata.get("status") != "trained":
        raise ValueError("Tune an untuned training artifact; don't retune a frozen candidate")
    records, scores, elapsed = score_records(model, manifest, "development")
    options = on_hits_options or contract["decision"]["on_hits_options"]
    if any(hit not in contract["decision"]["on_hits_options"] for hit in options):
        raise ValueError("Confirmation not prespecified by contract")
    trials = []
    for on_hits in options:
        for threshold in contract["decision"]["threshold_grid"]:
            decision = {
                "on_threshold": threshold,
                "off_threshold": threshold * contract["decision"]["off_threshold_ratio"],
                "on_hits": on_hits, "off_hits": contract["decision"]["off_hits"],
            }
            metrics = summarize([record_metrics(r, apply_decision(scores[r["id"]], decision), contract)
                                 for r in records], uncertainty=False)
            if metrics["event_recall"] is None or metrics["false_alarms_per_hour"] is None:
                raise ValueError("Development needs eligible positive events and verified negative exposure")
            trials.append({"decision": decision, "metrics": metrics})
    limit = contract["targets"]["false_alarms_per_hour_upper_95"]

    def rank(trial):
        m = trial["metrics"]
        delay = m["p95_delay_seconds_detected_events_only"]
        delay = delay if delay is not None else 1e12
        rate, recall = m["false_alarms_per_hour"], m["event_recall"]
        feasible = rate <= limit
        return (0, -recall, delay, rate, -trial["decision"]["on_threshold"]) if feasible else (
            1, rate, -recall, delay, -trial["decision"]["on_threshold"]
        )

    selected = min(trials, key=rank)
    metadata = {**model.metadata, "status": "frozen", "decision": selected["decision"],
                "contract": contract, "contract_sha256": digest(contract),
                "selection": {"split": "development", "manifest_sha256": file_hash(manifest),
                              "trials": trials, "chosen": selected,
                              "score_runtime_seconds": elapsed,
                              "criterion": contract["decision"]["selection"],
                              "warning": "Point-rate constraint selects a candidate; independent evidence and uncertainty are assessed separately."}}
    return save_model(output, metadata, model.arrays)


def target_assessment(summary, records, contract, runtime):
    targets = contract["targets"]
    evidence = {
        "authentic_only": all(r["origin"] == "authentic" for r in records),
        "independent_groups": summary["independent_groups"] >= targets["minimum_independent_groups"],
        "events": summary["events"] >= targets["minimum_events"],
        "negative_hours": summary["negative_hours"] >= targets["minimum_negative_hours"],
        "sources": len({r["source"] for r in records}) >= targets["minimum_sources"],
        "devices": len({r["device"] for r in records if r.get("device")}) >= targets["minimum_known_devices"],
        "drone_types": len({r["drone_type"] for r in records if r.get("drone_type")}) >= targets["minimum_known_drone_types"],
    }
    uncertainty = summary["uncertainty"]
    low, upper = uncertainty["recall_conservative_lower_95"], uncertainty["fa_per_hour_conservative_upper_95"]
    delay = summary["p95_delay_seconds_detected_events_only"]
    checks = {
        "recall": low is not None and low >= targets["event_recall_lower_95"],
        "false_alarms": upper is not None and upper <= targets["false_alarms_per_hour_upper_95"],
        "delay": delay is not None and delay <= targets["p95_detection_delay_seconds"],
        "runtime_on_reported_hardware": runtime <= targets["end_to_end_real_time_factor"],
    }
    return {"evidence_requirements": evidence, "numerical_checks": checks,
            "reliability_claim_supported": all(evidence.values()) and all(checks.values()),
            "status": "assessed on observed corpus only" if all(evidence.values()) else
            "INSUFFICIENT INDEPENDENT AUTHENTIC EVIDENCE; numerical results are diagnostic"}


def evaluate(bundles, manifest, split, output, allow_final_reuse=False):
    if split not in {"development", "final"}:
        raise ValueError("Evaluation is development or final only")
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite evaluation evidence")
    models = [load_model(bundle, require_frozen=True) for bundle in bundles]
    if not models:
        raise ValueError("Need at least one frozen model")
    rows = load_manifest(manifest, require_split=True)
    selected = [r for r in rows if r["split"] == split and r["annotation_status"] in VERIFIED]
    if not selected:
        raise ValueError("No verified evaluation records")
    for model in models:
        if model.metadata["training"]["manifest_sha256"] != file_hash(manifest):
            raise ValueError("Frozen manifest mismatch")
    if len({m.metadata["contract_sha256"] for m in models}) != 1:
        raise ValueError("Comparisons must use the same frozen contract")
    output.mkdir(parents=True)
    plan = {"models": [m.model_id for m in models], "manifest_sha256": file_hash(manifest),
            "split": split, "code": revision(),
            "contract_sha256": models[0].metadata["contract_sha256"],
            "evaluated_sha256": sorted({r["sha256"] for r in selected})}
    reused = False
    if split == "final":
        ledger = Path(manifest).with_suffix(".final-use.jsonl")
        history = read_jsonl(ledger) if ledger.exists() else []
        evaluated = set(plan["evaluated_sha256"])
        reused = any(evaluated.intersection(item["evaluated_sha256"]) for item in history)
        if reused and not allow_final_reuse:
            output.rmdir()
            raise ValueError("Final audio already evaluated. Further use is development evidence; disclose with --allow-final-reuse or obtain a fresh holdout")
        with ledger.open("a") as handle:
            handle.write(__import__("json").dumps({**plan, "reused": reused}) + "\n")
    # Written before any final waveform is scored: all comparison arms frozen.
    write_json(output / "frozen-plan.json", {**plan, "final_reused": reused})
    results = []
    for model in models:
        records, traces, elapsed = score_records(model, manifest, split)
        summaries = []
        contract = model.metadata["contract"]
        folder = output / model.model_id[:16]
        folder.mkdir()
        for row in records:
            trace = apply_decision(traces[row["id"]], model.metadata["decision"])
            summaries.append(record_metrics(row, trace, contract))
            # Filenames are hashes: annotation IDs cannot escape this folder.
            write_jsonl(folder / f"{digest(row['id'])[:20]}.scores.jsonl", trace)
        total = summarize(summaries)
        runtime = elapsed / sum(r["duration_seconds"] for r in records)
        compute = [item["compute_seconds"] for trace in traces.values() for item in trace]
        result = {
            "model_id": model.model_id, "decision": model.metadata["decision"],
            "features": model.metadata["features"], "summary": total,
            "source_breakdown": {source: summarize([r for r in summaries if r["source"] == source])
                                 for source in sorted({r["source"] for r in summaries})},
            "records": summaries, "runtime": {"hardware": hardware(), "seconds": elapsed,
                "real_time_factor": runtime, "p95_feature_score_decision_seconds":
                float(np.quantile(compute, .95)) if compute else None,
                "includes": "input checksum, decoding, resampling, rolling context, feature extraction, scoring, alarm logic; excludes one-time model import/load and JSON report output",
                "capture": "recorded replay, 100 ms canonical-rate delivery blocks; no live microphone measurement"},
            "assessment": target_assessment(total, records, contract, runtime),
        }
        if reused:
            result["assessment"]["reliability_claim_supported"] = False
            result["assessment"]["status"] = "REUSED FINAL SET: development evidence, requires fresh independent assessment"
        elif split == "development":
            result["assessment"]["reliability_claim_supported"] = False
            result["assessment"]["status"] = "DEVELOPMENT ONLY: not an independent final reliability assessment"
        write_json(folder / "results.json", result)
        results.append(result)
    report = {"schema_version": 1, "split": split, "final_reused": reused,
              "manifest_sha256": file_hash(manifest), "results": results}
    write_json(output / "comparison.json", report)
    return report
