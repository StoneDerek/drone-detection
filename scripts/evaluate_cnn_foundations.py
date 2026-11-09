#!/usr/bin/env python3
"""Full development traces, disclosed calibration and preserved model controls."""

import argparse
import gzip
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from drone_detector.common import canonical, digest, file_hash, read_json, write_json
from drone_detector.data import checked_audio, load_manifest
from drone_detector.model import load_model, save_model
from drone_detector.spectral_calibration import choose_budget_trial, exact_budget_trials, freeze_calibration_metadata
from drone_detector.spectrogram import SpectrogramConfig, axes, spectrogram, tensorflow
from drone_detector.stream import Alarm, replay_audio
from drone_detector.weak import recording_results, weak_metrics

ROOT = Path('runs/cnn-foundations-v1')
REPORT = Path('reports/cnn-foundations-v1')
MANIFEST = Path('runs/real-v2/corpus/manifest-v2.jsonl')
UNIFORM = Path('runs/spectral-lab-v1/cache/linear256-fft4096')
DENSE = ROOT / 'dense-development'


def prepare_dense():
    """Compute each full replay frontend once for all model comparisons."""
    DENSE.mkdir(parents=True, exist_ok=True)
    index_path = DENSE / 'index.json'
    config = SpectrogramConfig(axis='linear_filterbank', mel_bins=256)
    identity = {'manifest_sha256': file_hash(MANIFEST), 'features': config.to_dict(),
                'scope': 'DEVELOPMENT only; all replay contexts, shared float32 tensors'}
    if index_path.exists():
        index = read_json(index_path)
        if index['identity'] != identity:
            raise ValueError('Dense cache identity changed')
        return index
    records = [r for r in load_manifest(MANIFEST, True) if r['split'] == 'development']
    shape = (len(axes(config)[0]), len(axes(config)[1]))
    inventory = []
    placeholder = {'on_threshold': .5, 'off_threshold': .375, 'on_hits': 2, 'off_hits': 2}
    for number, row in enumerate(records):
        audio = checked_audio(row, MANIFEST)
        proxy = SimpleNamespace(config=config, model_id='template', metadata={'decision': placeholder}, score=lambda wave: .5)
        trace = replay_audio(proxy, audio)
        positions = [i for i, item in enumerate(trace) if item['reason'] == 'scored']
        cache = DENSE / (row['id']+'.npy')
        images = np.lib.format.open_memmap(cache, mode='w+', dtype=np.float32, shape=(len(positions), *shape))
        for i, position in enumerate(positions):
            start = round(trace[position]['window_start_seconds']*16000)
            images[i] = spectrogram(audio[start:start+48000], config)
        images.flush()
        del images
        template = [{k: v for k, v in item.items() if k not in {'score', 'alarm', 'model_id', 'compute_seconds', 'delivered_at_seconds'}} for item in trace]
        inventory.append({'id': row['id'], 'sha256': row['sha256'], 'cache': cache.name,
            'cache_sha256': file_hash(cache), 'positions': positions, 'template': template})
        if (number+1) % 20 == 0:
            print('Dense frontend', number+1, '/', len(records), flush=True)
    result = {'identity': identity, 'records': inventory}
    write_json(index_path, result)
    return result


def common_metrics(rows):
    labels = np.asarray([r['label'] == 'drone' for r in rows])
    losses = np.asarray([r['bag_loss'] for r in rows])
    coherent = np.asarray([r.get('consistent_max_loss', np.logaddexp(0, (1-2*int(label))*logit(np.clip(r['max_score'], 1e-7, 1-1e-7))))
                           for r, label in zip(rows, labels, strict=True)])
    return {'balanced_bag_loss': float((losses[labels].mean()+losses[~labels].mean())/2),
            'consistent_max_balanced_loss': float((coherent[labels].mean()+coherent[~labels].mean())/2),
            'max_context_auc': float(roc_auc_score(labels, [r['max_score'] for r in rows]))}


def uniform_training_report(model, name):
    original_name = name.removesuffix('-at8').removesuffix('-last')
    path = ROOT / 'trained' / (original_name+'-audit') / f"evaluation-{model.metadata['training']['selected_epoch']:02d}.json"
    if path.exists():
        return read_json(path)
    index = read_json(UNIFORM / 'index.json')
    summaries = {}
    for split in ['train', 'development']:
        rows = []
        for row in index['records']:
            if row['split'] != split:
                continue
            images = np.load(UNIFORM / row['cache'], mmap_mode='r', allow_pickle=False)
            scores = np.concatenate([model.score_images(images[i:i+16]) for i in range(0, len(images), 16)])
            logits = logit(np.clip(scores, 1e-7, 1-1e-7))
            loss = np.logaddexp(0, -logits.max()) if row['label'] == 'drone' else np.logaddexp(0, logits).mean()
            rows.append({k: row[k] for k in ['id', 'source', 'group_id', 'split', 'label']} | {
                'max_score': float(scores.max()), 'bag_loss': float(loss),
                'winning_start_sample': row['starts'][int(scores.argmax())]})
        summaries[split] = {**common_metrics(rows), 'records': rows}
    return summaries


def evaluate(name, index):
    output = ROOT / 'frozen' / name
    report_file = REPORT / (name+'-evaluation.json')
    if output.exists():
        if not report_file.exists():
            raise ValueError('Frozen model exists without its evaluation report')
        print('Preserving completed evaluation', name, flush=True)
        return
    model = load_model(ROOT / 'trained' / name)
    records = [r for r in load_manifest(MANIFEST, True) if r['split'] == 'development']
    by_id = {row['id']: row for row in records}
    started = time.perf_counter()
    trace_dir = ROOT / 'traces'
    trace_dir.mkdir(exist_ok=True)
    trace_file = trace_dir / (name+'.jsonl.gz')
    scores, parity, dense_bags = {}, None, []
    with gzip.open(trace_file, 'wb') as stream:
        for number, cached in enumerate(index['records']):
            row = by_id[cached['id']]
            if file_hash(DENSE / cached['cache']) != cached['cache_sha256']:
                raise ValueError('Dense tensor changed')
            images = np.load(DENSE / cached['cache'], mmap_mode='r', allow_pickle=False)
            values = np.concatenate([model.score_images(images[i:i+16]) for i in range(0, len(images), 16)]) if len(images) else np.empty(0)
            trace = [dict(item) for item in cached['template']]
            for item in trace:
                item.update({'score': None if item['reason'] == 'insufficient_audio' else 0.,
                    'model_id': model.model_id, 'compute_seconds': None, 'delivered_at_seconds': None,
                    'processing_mode': 'offline shared causal tensors; no delivery-time claim'})
            for position, value in zip(cached['positions'], values, strict=True):
                trace[position]['score'] = float(value)
            # Raw traces carry a disclosed placeholder decision; calibrated
            # recording outcomes below reapply the chosen frozen decision.
            alarm = Alarm({'on_threshold': .5, 'off_threshold': .375, 'on_hits': 2, 'off_hits': 2})
            for item in trace:
                item['alarm'] = alarm.step(item['score'])
            scores[row['id']] = trace
            stream.write(canonical({'recording_id': row['id'], 'trace': trace})+b'\n')
            if values.size:
                logits = logit(np.clip(values, 1e-7, 1-1e-7))
                loss = np.logaddexp(0, -logits.max()) if row['label'] == 'drone' else np.logaddexp(0, logits).mean()
                dense_bags.append({'id': row['id'], 'label': row['label'], 'max_score': float(values.max()), 'bag_loss': float(loss)})
            if number == 0:
                audio = checked_audio(row, MANIFEST)
                sequential = replay_audio(model, audio, {'on_threshold': .5, 'off_threshold': .375, 'on_hits': 2, 'off_hits': 2})
                assert len(sequential) == len(trace)
                maximum = max(abs(a['score']-b['score']) for a, b in zip(sequential, trace, strict=True) if a['score'] is not None)
                assert maximum <= 1e-5
                parity = {'recording_id': row['id'], 'windows': len(trace), 'maximum_score_error': maximum, 'passed': True}
            if (number+1) % 40 == 0:
                print('Evaluate', name, number+1, '/', len(records), flush=True)
    trials = exact_budget_trials(records, scores)
    chosen = choose_budget_trial(trials)
    decision = chosen['decision']
    outcomes = recording_results(records, scores, decision)
    breakdown = {source: weak_metrics([row for row in outcomes if row['source'] == source]) for source in sorted({r['source'] for r in records})}
    contract = {'scope': 'DEVELOPMENT calibration only;5% negative-recording budget, no independent field-FPR claim',
                'negative_recordings': 188, 'maximum_alarm_files': 9, 'on_hits_candidates': [1, 2],
                'numerical_margin': 1e-5, 'native_decision_frozen': True}
    meta = freeze_calibration_metadata(model.metadata, decision, contract,
        {'split': 'development', 'chosen': chosen, 'trials': trials, 'contract_sha256': digest(contract)})
    # The initial plan's code identifies protocol preparation. Preserve that
    # identity and attach the exact implementation snapshot used to execute
    # each arm before creating its calibrated package.
    stem = name.removesuffix('-at8').removesuffix('-last')
    if stem.startswith('full-recording-'):
        execution = read_json(REPORT / 'full-recording-plan.json')['code']
        archive = ROOT / 'full-recording-implementation.tar.gz'
    elif stem.startswith('coherent-'):
        execution = read_json(REPORT / 'coherent-followup-plan.json')['code']
        archive = ROOT / 'coherent-implementation.tar.gz'
    elif stem in {'medium-sampled-max', 'medium-full-smooth'}:
        execution = read_json(REPORT / 'implementation-training-initial.json')['code']
        archive = ROOT / 'training-implementation-initial.tar.gz'
    elif stem == 'medium-balanced-max':
        execution = read_json(REPORT / 'balanced-followup-plan.json')['code']
        archive = ROOT / 'balanced-implementation.tar.gz'
    elif stem in {'medium-balanced-global-head', 'medium-balanced-frequency-head', 'medium-full-max'}:
        kind = 'full-max' if stem == 'medium-full-max' else 'heads'
        execution = read_json(REPORT / f'{kind}-followup-plan.json')['code']
        archive = ROOT / f'{kind}-implementation.tar.gz'
    else:
        execution = read_json(REPORT / 'plan.json')['code']
        archive = ROOT / 'initial-source.tar.gz'
    meta.update({'protocol_preparation_code': meta['code'], 'code': execution,
                 'training_implementation_archive_sha256': file_hash(archive)})
    frozen = save_model(output, meta, model.arrays)
    uniform = uniform_training_report(model, name)
    report = {'name': name, 'parent_model_id': model.model_id, 'model_id': frozen.model_id,
        'architecture': model.metadata['architecture'],
        'training': {k: v for k, v in model.metadata['training'].items() if k != 'records'},
        'development': chosen, 'source_breakdown': breakdown,
        'uniform_context_evaluation': uniform, 'dense_context_evaluation': common_metrics(dense_bags),
        'seconds': time.perf_counter()-started, 'batch_sequential_parity': parity,
        'raw_traces': str(trace_file), 'raw_trace_sha256': file_hash(trace_file),
        'raw_trace_decision': 'placeholder.5; calibrated outcomes reapply frozen decision',
        'recordings': [{k: v for k, v in row.items() if k != 'trace'} for row in outcomes]}
    write_json(report_file, report)
    print('Frozen development result', name, chosen, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--models', nargs='*')
    args = parser.parse_args()
    tf = tensorflow()
    tf.config.threading.set_intra_op_parallelism_threads(1)
    index = prepare_dense()
    if args.prepare_only:
        return
    names = args.models or [p.name for p in sorted((ROOT / 'trained').iterdir()) if (p / 'model.json').exists()]
    for name in names:
        evaluate(name, index)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
