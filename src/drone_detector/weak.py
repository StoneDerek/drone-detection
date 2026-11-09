"""Learn from publisher-level presence labels without inventing event boundaries.

EM-style multiple-instance learning: select the most positive instance in a
positive bag; retain all sampled negative instances. This is experimental weak
supervision, not a claim that every positive-file window contains a drone.
"""

import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .audio import FeatureConfig, features
from .common import digest, file_hash, hardware, revision, versions, write_json, write_jsonl
from .data import checked_audio, load_manifest
from .model import load_model, save_model
from .stream import alarm_intervals, replay_audio


def bags_for_training(manifest, config, encoder=None):
    records = [r for r in load_manifest(manifest, True) if r["split"] == "train"
               and r["label"] in {"drone", "non_drone"}]
    bags, used = [], []
    for record in records:
        audio = checked_audio(record, manifest, config.sample_rate)
        starts = list(range(0, max(1, len(audio) - config.window_samples + 1), config.hop_samples))
        # Cover each original uniformly instead of keeping its first 250 ms.
        # Explicit cap keeps long files from dominating compute or sample count.
        if len(starts) > 60:
            starts = [starts[i] for i in np.linspace(0, len(starts) - 1, 60).astype(int)]
        vectors = []
        for start in starts:
            chunk = audio[start:start + config.window_samples]
            if len(chunk) / config.sample_rate < config.min_audio_seconds:
                continue
            if np.sqrt(np.mean(chunk.astype(np.float64)**2)) < config.silence_rms:
                continue
            vectors.append(features(chunk, config, encoder))
        if vectors:
            bags.append(np.asarray(vectors))
            used.append(record)
    if {r["label"] for r in used} != {"drone", "non_drone"}:
        raise ValueError("Both presence and absence training bags are required")
    return bags, used


def train_weak(manifest, output, representation="relative_mel", seed=42, c=1.0, encoder_source=None):
    begin = time.perf_counter()
    config = FeatureConfig(representation=representation)
    encoder = None
    if representation.startswith('yamnet'):
        if encoder_source is None:
            raise ValueError("YAMNet training requires its frozen official encoder")
        from .encoder import YAMNet
        encoder = YAMNet(encoder_source)
    bags, records = bags_for_training(manifest, config, encoder)
    y = np.array([int(r["label"] == "drone") for r in records])
    scaler = StandardScaler()
    all_x = np.concatenate(bags)
    record_weights = np.concatenate([np.full(len(bag), 1 / len(bag)) for bag in bags])
    scaler.fit(all_x, sample_weight=record_weights)
    scaled = [scaler.transform(bag) for bag in bags]
    classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed)
    classifier.fit(np.array([bag.mean(axis=0) for bag in scaled]), y)
    selected = None
    for _ in range(4):
        x, target, weight = [], [], []
        selected = []
        for bag, label in zip(scaled, y, strict=True):
            if label:
                index = int(np.argmax(classifier.decision_function(bag)))
                chosen = bag[index:index + 1]
                selected.append(index)
            else:
                chosen = bag
                selected.append(None)
            x.extend(chosen)
            target.extend([label] * len(chosen))
            weight.extend([1 / len(chosen)] * len(chosen))
        weight, target = np.asarray(weight), np.asarray(target)
        for label in [0, 1]:
            weight[target == label] *= len(target) / (2 * weight[target == label].sum())
        classifier = LogisticRegression(C=c, max_iter=2000, random_state=seed)
        classifier.fit(np.asarray(x), target, sample_weight=weight)
        if int(classifier.n_iter_[0]) >= 2000:
            raise ValueError("Weak classifier failed to converge")
    metadata = {
        "schema_version": 1, "status": "trained", "features": config.to_dict(),
        "class_mapping": {"0": "non_drone", "1": "drone"},
        "feature_dimension": len(scaler.mean_), "code": revision(), "versions": versions(),
        "training": {"manifest_sha256": file_hash(manifest), "seed": seed,
            "method": "MI logistic regression; initialize bag means, four train-only max-positive-instance updates; all uniformly sampled negative windows, equal bag mass then balanced classes",
            "label_scope": "Publisher clip-level labels, not independently verified event boundaries",
            "C": c, "max_uniform_windows_per_recording": 60,
            "seconds": time.perf_counter() - begin,
            "origins": sorted({r["origin"] for r in records}),
            "records": [{"id": r["id"], "group_id": r["group_id"], "source": r["source"],
                         "sha256": r["sha256"], "label": r["label"], "sampled_windows": len(bag),
                         "positive_instance_index": index}
                        for r, bag, index in zip(records, bags, selected, strict=True)]},
        "normalization": "Weighted train-bag windows only; mean/scale stored in NPZ",
        "limitations": "Research prototype trained on real publisher-labeled audio. Event recall/delay and field reliability remain unvalidated. Scores are uncalibrated; unfamiliar sounds can trigger alarms.",
    }
    if encoder is not None:
        metadata["encoder"] = {"name": "Google YAMNet v1, AudioSet-pretrained MobileNet",
            "source": "https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed",
            "license": "Apache-2.0 per published model card",
            "input_spec": "16 kHz mono float waveform [-1,1]; encoder's own 64-band log-Mel/0.96 s patches, mean of 1024-D patch embeddings per 3 s context",
            "pretraining_overlap": "AudioSet/YouTube and public-source overlap not ruled out; external capture independence remains limited"}
    if representation == 'yamnet_rms':
        metadata['experimental'] = True
        metadata['input_gain'] = {'scope': 'available samples in current context, before padding',
                                  'target_rms': config.gain_target_rms, 'maximum_gain_db': config.gain_max_db,
                                  'peak_cap': .98, 'silence_guard_unchanged': True}
        metadata['limitations'] = 'DEVELOPMENT-ONLY gain-normalization experiment. No fresh independent assessment; not a replacement for the assessed default model.'
    return save_model(output, metadata, {"mean": scaler.mean_, "scale": scaler.scale_,
                                       "coef": classifier.coef_[0], "intercept": classifier.intercept_},
                      encoder_source=encoder_source)


def score_split(model, manifest, split):
    if model.metadata["training"]["manifest_sha256"] != file_hash(manifest):
        raise ValueError("Manifest differs from frozen data identity")
    records = [r for r in load_manifest(manifest, True) if r["split"] == split
               and r["label"] in {"drone", "non_drone"}]
    if not records:
        raise ValueError("No labeled recordings in split")
    scores, elapsed = {}, 0.0
    placeholder = {"on_threshold": .5, "off_threshold": .375, "on_hits": 1, "off_hits": 2}
    for record in records:
        begin = time.perf_counter()
        audio = checked_audio(record, manifest, model.config.sample_rate)
        scores[record["id"]] = replay_audio(model, audio, placeholder)
        elapsed += time.perf_counter() - begin
    return records, scores, elapsed


def recording_results(records, scores, decision):
    from .evaluation import apply_decision

    result = []
    for record in records:
        trace = apply_decision(scores[record["id"]], decision)
        alarms = alarm_intervals(trace, record["duration_seconds"], merge_gap=1)
        result.append({"id": record["id"], "source": record["source"],
            "group_id": record["group_id"], "label": record["label"],
            "duration_seconds": record["duration_seconds"],
            "truth": int(record["label"] == "drone"), "predicted": int(bool(alarms)),
            "max_score": max((row["score"] or 0 for row in trace), default=0),
            "alarms": alarms, "trace": trace})
    return result


def weak_metrics(rows, intervals=False):
    y = np.array([r["truth"] for r in rows])
    pred = np.array([r["predicted"] for r in rows])
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    tpr = tp / (tp + fn) if tp + fn else None
    fpr = fp / (tn + fp) if tn + fp else None
    result = {"recordings": len(rows), "provenance_groups": len({r["group_id"] for r in rows}),
        "positive_recordings": int(y.sum()), "negative_recordings": int((1 - y).sum()),
        "positive_hours": sum(r["duration_seconds"] for r in rows if r["truth"]) / 3600,
        "negative_hours": sum(r["duration_seconds"] for r in rows if not r["truth"]) / 3600,
        "recording_recall": tpr, "negative_recording_false_positive_fraction": fpr,
        "precision": tp / (tp + fp) if tp + fp else None,
        "balanced_accuracy": (tpr + 1 - fpr) / 2 if tpr is not None and fpr is not None else None,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "roc_auc_max_window_score": float(roc_auc_score(y, [r["max_score"] for r in rows]))
        if len(set(y)) == 2 else None,
        "average_precision_max_window_score": float(average_precision_score(y, [r["max_score"] for r in rows]))
        if len(set(y)) == 2 else None,
        "event_metrics": "Not assessed: publisher clip labels do not establish temporal truth",
        "field_reliability_claim": False}
    if intervals:
        groups = sorted({r["group_id"] for r in rows})
        by_group = {g: [r for r in rows if r["group_id"] == g] for g in groups}
        recalls, rates = [], []
        rng = np.random.default_rng(20261108)
        for _ in range(2000):
            sampled = [r for group in rng.choice(groups, len(groups)) for r in by_group[group]]
            positive = [r for r in sampled if r["truth"]]
            negative = [r for r in sampled if not r["truth"]]
            if positive:
                recalls.append(sum(r["predicted"] for r in positive) / len(positive))
            if negative:
                rates.append(sum(r["predicted"] for r in negative) / len(negative))
        result["group_bootstrap_95"] = {
            "recall": [float(x) for x in np.quantile(recalls, [.025, .975])] if recalls else None,
            "false_positive_fraction": [float(x) for x in np.quantile(rates, [.025, .975])] if rates else None,
            "warning": "Conservative source/day groups, not independent windows. Few positive-source groups make intervals inadequate for field claims."}
    return result


def tune_weak(bundle, manifest, output, policy="v1"):
    model = load_model(bundle)
    if model.metadata["status"] != "trained":
        raise ValueError("Cannot retune an already frozen candidate")
    records, scores, elapsed = score_split(model, manifest, "development")
    trials = []
    for hits in [1, 2]:
        for threshold in [.2, .35, .5, .65, .8, .9, .95, .98, .995]:
            decision = {"on_threshold": threshold, "off_threshold": .75 * threshold,
                        "on_hits": hits, "off_hits": 2}
            result = weak_metrics(recording_results(records, scores, decision))
            if result["recording_recall"] is None or result["negative_recording_false_positive_fraction"] is None:
                raise ValueError("Need both classes in development")
            trials.append({"decision": decision, "metrics": result})

    def rank(trial):
        m = trial["metrics"]
        fpr, recall = m["negative_recording_false_positive_fraction"], m["recording_recall"]
        if policy == 'sensitivity-v2':
            return (fpr > .05, -recall if fpr <= .05 else fpr,
                    trial['decision']['on_threshold'] if fpr <= .05 else -recall,
                    fpr, -trial['decision']['on_hits'])
        return (fpr > .05, -recall if fpr <= .05 else fpr, fpr if fpr <= .05 else -recall,
                -trial["decision"]["on_threshold"], trial["decision"]["on_hits"])

    chosen = min(trials, key=rank)
    total_windows = sum(len(trace) for trace in scores.values())
    quality_coverage = {
        'recordings': len(records), 'windows': total_windows,
        'low_frequency_flagged_windows': sum('low_frequency_dominance' in row.get('quality_flags', []) for trace in scores.values() for row in trace),
        'clipping_flagged_windows': sum('clipping' in row.get('quality_flags', []) for trace in scores.values() for row in trace),
        'interpretation': 'Uncalibrated input advisories, not measured wind/SNR or an accuracy/coverage improvement. Binary alarm decisions unchanged by flags.'}
    selection = 'Development only: prefer negative-file FPR <=5%, highest recall, lowest FPR, highest threshold, lowest confirmation count. Numerical choices are provisional.'
    if policy == 'sensitivity-v2':
        selection = 'Development only: prefer negative-file FPR <=5%, highest recall, lowest threshold meeting the budget, then lowest FPR and two-hit confirmation. Provisional sensitivity policy addresses long-file extreme-score bias.'
    contract = {"id": "publisher-presence-prototype-" + policy, "label_scope": "weak recording-level publisher labels",
                "selection": selection,
                "event_reliability_contract": "contracts/evaluation-v1.json remains unfulfilled",
                "no_field_reliability_claim": True}
    metadata = {**model.metadata, "status": "frozen", "decision": chosen["decision"],
                "contract": contract, "contract_sha256": digest(contract),
                "selection": {"split": "development", "trials": trials, "chosen": chosen,
                              'quality_advisory_coverage': quality_coverage,
                              "runtime_seconds": elapsed, "code": revision()}}
    encoder_source = Path(bundle) / "encoder" if model.encoder is not None else None
    return save_model(output, metadata, model.arrays, encoder_source)


def evaluate_weak(bundles, manifest, split, output):
    output = Path(output)
    if output.exists() or split not in {"development", "final"}:
        raise ValueError("Use a fresh output and development/final split")
    models = [load_model(bundle, True) for bundle in bundles]
    if any(m.metadata["training"]["manifest_sha256"] != file_hash(manifest) for m in models):
        raise ValueError("Manifest identity changed")
    final_use = Path(manifest).with_suffix(".weak-final-use.json")
    if split == "final" and final_use.exists():
        raise ValueError("Final set already assessed; further optimization needs a fresh assessment")
    output.mkdir(parents=True)
    plan = {"models": [m.model_id for m in models], "manifest_sha256": file_hash(manifest),
            "code": revision(), "split": split, "label_scope": "publisher weak clip labels; no event truth"}
    write_json(output / "frozen-plan.json", plan)
    if split == "final":
        write_json(final_use, plan)
    results = []
    for model in models:
        records, scores, elapsed = score_split(model, manifest, split)
        rows = recording_results(records, scores, model.metadata["decision"])
        folder = output / model.model_id[:16]
        folder.mkdir()
        for row in rows:
            write_jsonl(folder / f"{digest(row['id'])[:20]}.scores.jsonl", row.pop("trace"))
        result = {"model_id": model.model_id, "decision": model.metadata["decision"],
            "representation": model.config.representation, "summary": weak_metrics(rows, True),
            "source_breakdown": {s: weak_metrics([r for r in rows if r["source"] == s], True)
                                 for s in sorted({r["source"] for r in rows})},
            "records": rows, "runtime": {"seconds": elapsed,
                "real_time_factor": elapsed / sum(r["duration_seconds"] for r in rows),
                "hardware": hardware()}, "field_reliability_claim": False}
        write_json(folder / "results.json", result)
        results.append(result)
    report = {**plan, "results": results}
    write_json(output / "comparison.json", report)
    return report
