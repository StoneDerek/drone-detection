#!/usr/bin/env python3
"""Fixed classifiers on other conditions and a separate fresh positive cohort."""

import argparse
import gzip
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from drone_detector.audio import decode
from drone_detector.common import canonical, file_hash, read_json, read_jsonl, write_json, write_jsonl
from drone_detector.model import load_model
from drone_detector.spectrogram import SpectrogramConfig, spectrogram
from drone_detector.stream import replay_audio

REPORT = Path('reports/cross-condition-v1')


def context_starts(length, maximum=60):
    starts = np.arange(0, max(0, length - 16000 + 1), 8000, dtype=np.int64)
    if maximum is not None and len(starts) > maximum:
        starts = starts[np.linspace(0, len(starts) - 1, maximum).astype(int)]
    return starts


def fixed_models(plan):
    models = {}
    for arm in plan['models']:
        root = Path(arm['bundle'])
        if file_hash(root / 'model.json') != arm['metadata_sha256'] or file_hash(root / 'parameters.npz') != arm['parameters_sha256']:
            raise ValueError('A fixed model package changed')
        if arm['threshold'] != .5:
            raise ValueError('The prespecified classifier threshold changed')
        if arm['family'] == 'cnn':
            model = load_model(root)
            if model.model_id != arm['model_id']:
                raise ValueError('Classifier identity changed')
        else:
            with np.load(root / 'parameters.npz', allow_pickle=False) as archive:
                model = {key: archive[key].copy() for key in archive.files}
            if set(model) != {'mean', 'scale', 'coef', 'intercept'} or np.any(model['scale'] <= 0):
                raise ValueError('Invalid fixed non-neural package')
            if any(model[key].shape != (512,) for key in ['mean', 'scale', 'coef']) or model['intercept'].shape != (1,) or not all(np.isfinite(value).all() for value in model.values()):
                raise ValueError('Invalid fixed non-neural dimensions or values')
        models[arm['name']] = model
    return models


def source_metrics(rows):
    result = {}
    for source in sorted({row['source'] for row in rows}):
        selected = [row for row in rows if row['source'] == source]
        positive = [row for row in selected if row['label'] == 'drone']
        negative = [row for row in selected if row['label'] == 'non_drone']
        eligible = [row for row in selected if row['sampled_contexts']]
        result[source] = {'recordings': len(selected), 'conservative_groups': len({row['group_id'] for row in selected}),
            'positive_files': len(positive), 'negative_files': len(negative),
            'positive_files_detected': sum(row['predicted'] for row in positive),
            'negative_files_classified_positive': sum(row['predicted'] for row in negative),
            'insufficient_full_context_files': len(selected) - len(eligible),
            'sampled_contexts': sum(row['sampled_contexts'] for row in selected),
            'silence_guard_contexts': sum(row['silent_contexts'] for row in selected),
            'max_score_auc': float(roc_auc_score([row['label'] == 'drone' for row in eligible], [row['max_score'] for row in eligible]))
            if eligible and {row['label'] for row in eligible} == {'drone', 'non_drone'} else None,
            'scope': 'Recording-bag publisher labels. Sampled seconds are not event truth or independent trials.'}
    return result


def transfer_records():
    old = Path('runs/real-v2/corpus/manifest-v2.jsonl')
    records = []
    for row in read_jsonl(old):
        if row['split'] in ['train', 'development']:
            records.append({**row, 'path': str(old.parent / row['path']), 'prior_role': row['split']})
    records += [{**row, 'prior_role': 'consumed external-v3'} for row in read_jsonl('reports/external-v3/recordings.jsonl') if row['source'] == 'saarmets-fpv']
    if len(records) != 652:
        raise ValueError('Prespecified transfer cohort changed')
    write_jsonl(REPORT / 'transfer-recordings.jsonl', records)
    return records


def evaluate_phase(plan, models, records, phase):
    destination = REPORT / phase
    ledger_path = REPORT / f'{phase}-use.json'
    if (destination / 'comparison.json').exists():
        raise ValueError('Completed cohort preserved; reapplying retained scores is not fresh evidence')
    execution = {'plan_sha256': file_hash(REPORT / 'plan.json'), 'runner_sha256': file_hash(__file__),
                 'source_freeze_sha256': file_hash(REPORT / 'source-freeze.json'),
                 'models': [{k: v for k, v in arm.items() if k in ['name', 'model_id', 'metadata_sha256', 'parameters_sha256']} for arm in plan['models']],
                 'recording_identity': [{'id': row['id'], 'sha256': row['sha256']} for row in records],
                 'phase': phase, 'policy': 'Any sampled full1s classifier score>=0.5; no per-second ground-truth inference.',
                 'scope': 'Consumed cross-condition diagnosis' if phase == 'transfer' else 'Fresh human-primary-positive clips only, one apparent original group; no negative/event/range claim.'}
    if ledger_path.exists():
        if read_json(ledger_path) != execution:
            raise ValueError('Cannot resume a consumed cohort after changing its execution')
    else:
        write_json(ledger_path, execution)
    destination.mkdir(exist_ok=True)
    arms_by_rule = {}
    for arm in plan['models']:
        name = arm['name'].split('-seed')[0].removesuffix('-logistic')
        arms_by_rule.setdefault(name, []).append(arm)
    report = {'execution': execution, 'models': {}}
    for rule, arms in arms_by_rule.items():
        result_path = destination / f'{rule}.json'
        if result_path.exists():
            completed = read_json(result_path)
            report['models'].update(completed['models'])
            continue
        config = SpectrogramConfig(**arms[0]['features'])
        for arm in arms:
            if arm['features'] != arms[0]['features']:
                raise ValueError('Shared input rule mismatch')
        summaries = {arm['name']: [] for arm in arms}
        started = time.perf_counter()
        with gzip.open(destination / f'{rule}.scores.jsonl.gz', 'wb') as stream:
            for index, row in enumerate(records):
                if file_hash(row['path']) != row['sha256']:
                    raise ValueError('Pinned recording bytes changed')
                wave, _ = decode(row['path'], 16000)
                starts = context_starts(len(wave), 60 if phase == 'transfer' else None)
                images = np.stack([spectrogram(wave[start:start + 16000], config) for start in starts]) if len(starts) else np.empty((0, 256, 47), dtype=np.float32)
                rms = np.array([np.sqrt(np.mean(wave[start:start + 16000].astype(np.float64) ** 2)) for start in starts])
                per_model = {}
                for arm in arms:
                    model = models[arm['name']]
                    if arm['family'] == 'cnn':
                        scores = model.score_images(images) if len(images) else np.empty(0)
                        if index == 0 and len(scores):
                            sequential = model.score(wave[starts[0]:starts[0] + 16000])
                            if abs(float(scores[0]) - sequential) > 1e-5:
                                raise ValueError('Direct and batched classifier scores disagree')
                    else:
                        descriptors = np.concatenate([images.mean(axis=2), images.std(axis=2)], axis=1).astype(np.float64)
                        scores = expit(((descriptors - model['mean']) / model['scale']) @ model['coef'] + model['intercept'][0])
                    per_model[arm['name']] = scores.astype(float).tolist()
                    summaries[arm['name']].append({'id': row['id'], 'source': row['source'], 'group_id': row['group_id'],
                        'label': row['label'], 'sampled_contexts': len(starts), 'silent_contexts': int(np.sum(rms < config.silence_rms)),
                        'max_score': float(scores.max()) if len(scores) else None,
                        'positive_context_fraction': float(np.mean(scores >= .5)) if len(scores) else None,
                        'predicted': int(bool(len(scores) and np.any(scores >= .5)))})
                stream.write(canonical({'recording_id': row['id'], 'start_samples': starts.tolist(), 'rms': rms.tolist(), 'scores': per_model}) + b'\n')
                if (index + 1) % 100 == 0:
                    print(phase, rule, index + 1, '/', len(records), 'seconds', round(time.perf_counter() - started, 1), flush=True)
        completed = {'rule': rule, 'seconds': time.perf_counter() - started, 'models': {}}
        for arm in arms:
            name = arm['name']
            result = {'by_source': source_metrics(summaries[name]), 'recordings': summaries[name]}
            completed['models'][name] = result
            report['models'][name] = result
        write_json(result_path, completed)
        print('COMPLETE', phase, rule, flush=True)
    if phase == 'kenya':
        source = read_json(REPORT / 'source-freeze.json')
        if source['legacy_panel_sha256'] != file_hash('reports/external-v3/panel-plan.json'):
            raise ValueError('Frozen legacy panel changed')
        legacy = {}
        for arm in read_json('reports/external-v3/panel-plan.json')['models']:
            root = Path(arm['bundle'])
            if file_hash(root / 'model.json') != arm['model_json_sha256']:
                raise ValueError('Legacy package changed')
            model = load_model(root, require_frozen=True)
            result = []
            with gzip.open(destination / f'legacy-{arm["name"]}.jsonl.gz', 'wb') as stream:
                for row in records:
                    wave, _ = decode(row['path'], 16000)
                    trace = replay_audio(model, wave)
                    result.append({'id': row['id'], 'source': row['source'], 'group_id': row['group_id'], 'label': row['label'],
                                   'predicted': int(any(t['alarm'] for t in trace)), 'max_score': max(t['score'] or 0 for t in trace)})
                    stream.write(canonical({'recording_id': row['id'], 'trace': trace}) + b'\n')
            legacy[arm['name']] = {'decision': arm['decision'], 'recordings': result,
                'positive_files_detected': sum(row['predicted'] for row in result),
                'scope': 'Original3s replay and alarm, distinct from the1s classifier max rule.'}
        report['legacy_original_policy'] = legacy
    write_json(destination / 'comparison.json', report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['transfer', 'kenya'])
    args = parser.parse_args()
    plan = read_json(REPORT / 'plan.json')
    models = fixed_models(plan)
    if args.phase == 'transfer':
        records = transfer_records()
    else:
        audit = read_json(REPORT / 'kenya-data-audit.json')
        if audit['cross_corpus_whole_file_matches']:
            raise ValueError('Resolve cross-corpus overlap before fresh scores')
        records = read_jsonl(REPORT / 'kenya-recordings.jsonl')
    evaluate_phase(plan, models, records, args.phase)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
