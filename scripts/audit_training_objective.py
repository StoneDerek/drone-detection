#!/usr/bin/env python3
"""Audit full-recording targets against the actual cyclic sampled sub-bags."""

import math
from pathlib import Path

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from drone_detector.common import file_hash, read_json, write_json
from drone_detector.model import load_model


def main():
    out = Path('reports/training-audit-v1')
    cache = Path('runs/spectral-lab-v1/cache/linear256-fft4096')
    index = read_json(cache / 'index.json')
    model = load_model('artifacts/cnn-foundations-v1/frozen/coherent-frequency-seed42', True)
    net = model.network()
    results = []
    rng = np.random.default_rng(42)
    records = [row for row in index['records'] if row['split'] == 'train']
    for row in records:
        images = np.load(cache / row['cache'], mmap_mode='r', allow_pickle=False)
        values = np.concatenate([net(np.asarray(images[i:i+16])[..., None], training=False).numpy().ravel()
                                 for i in range(0, len(images), 16)])
        permutation = rng.permutation(len(values))
        count = min(8, len(values))
        offsets = range(0, math.lcm(len(values), count), count)
        submax = np.asarray([values[permutation[(offset+np.arange(count)) % len(values)]].max()
                             for offset in offsets])
        label = row['label'] == 'drone'
        loss = lambda x: np.logaddexp(0, -x if label else x)  # noqa: E731
        results.append({k: row[k] for k in ['id', 'source', 'label']} | {
            'contexts': len(values), 'distinct_cyclic_subbags': len(submax),
            'full_bag_loss': float(loss(values.max())), 'mean_sampled_subbag_loss': float(loss(submax).mean()),
            'full_max_score': float(expit(values.max())),
            'subbags_missing_all_full_maxima_fraction': float(np.mean(submax < values.max())),
            'subbags_below_drone_threshold_fraction': float(np.mean(expit(submax) < model.metadata['decision']['on_threshold'])),
            'qualification': 'Frozen model scores diagnose the objective discrepancy; they are not audible-window labels.'})
    summary = {}
    for label in ['drone', 'non_drone']:
        rows = [row for row in results if row['label'] == label]
        summary[label] = {'recordings': len(rows),
            'mean_full_recording_loss': float(np.mean([row['full_bag_loss'] for row in rows])),
            'mean_cyclic_sampled_loss': float(np.mean([row['mean_sampled_subbag_loss'] for row in rows])),
            'mean_fraction_subbags_missing_full_maximum': float(np.mean([row['subbags_missing_all_full_maxima_fraction'] for row in rows]))}
    n, m = 60, 8
    positive_full = float(np.logaddexp(0, -4.))
    positive_sampled = float(m/n*np.logaddexp(0, -4.)+(1-m/n)*np.logaddexp(0, 4.))
    negative_full = float(np.logaddexp(0, 4.))
    negative_sampled = float(m/n*np.logaddexp(0, 4.)+(1-m/n)*np.logaddexp(0, -4.))
    write_json(out / 'sampled-objective.json', {
        'scope': 'Fixed current weights and existing TRAIN inputs only; no new acoustic assessment',
        'model_id': model.model_id, 'manifest_sha256': index['identity']['manifest_sha256'],
        'script_sha256': file_hash(__file__),
        'witness': {'contexts': n, 'sampled_contexts': m, 'logits': 'one +4 and59 -4',
                    'full_positive_loss': positive_full, 'expected_sampled_positive_loss': positive_sampled,
                    'full_negative_loss': negative_full, 'expected_sampled_negative_loss': negative_sampled,
                    'rare_maximum_omission_probability': 1-m/n},
        'finding': 'BCE(max(sample8)) is not an unbiased estimate of BCE(max(full recording)). Positive sampling can omit the event required by its bag label and penalize background-only sub-bags; negative sampling weakens the penalty on rare maxima. Correcting positive/negative pooling symmetry did not remove this second discrepancy.',
        'summary': summary, 'recordings': results})
    print(summary, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
