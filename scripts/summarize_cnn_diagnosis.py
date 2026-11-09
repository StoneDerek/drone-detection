#!/usr/bin/env python3
"""Summarize controlled runs without tuning alarm thresholds."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from drone_detector.common import read_json, write_json
from drone_detector.model import load_model


def main():
    root = Path('runs/cnn-diagnosis-v1')
    report = Path('reports/cnn-diagnosis-v1')
    result = {'scope': 'Existing train/development sampled bags only, no new alarm calibration or independent assessment', 'arms': {}}
    for name in ['peak-relative', 'fixed-reference', 'rms-fixed-reference']:
        model = load_model(root / 'normalization-models' / name)
        cache = Path('runs/spectral-lab-v1/cache/linear256-fft4096') if name == 'peak-relative' else root / 'cache' / name
        index = read_json(cache / 'index.json')
        rows = []
        for row in index['records']:
            images = np.load(cache / row['cache'], mmap_mode='r', allow_pickle=False)
            scores = np.concatenate([model.score_images(images[start:start+16]) for start in range(0, len(images), 16)])
            logits = logit(np.clip(scores, 1e-7, 1-1e-7))
            positive = row['label'] == 'drone'
            rows.append({k: row[k] for k in ['id', 'source', 'split', 'label']} | {
                'sampled_contexts': len(scores), 'max_score': float(scores.max()),
                'bag_loss': float(np.logaddexp(0, -logits.max()) if positive else np.logaddexp(0, logits).mean()),
                'mean_input': float(images.mean()), 'max_score_context_start_sample': row['starts'][int(scores.argmax())]})
        summary = {}
        for split in ['train', 'development']:
            chosen = [r for r in rows if r['split'] == split]
            positive = np.asarray([r['label'] == 'drone' for r in chosen])
            loss = np.asarray([r['bag_loss'] for r in chosen])
            summary[split] = {'balanced_bag_loss': float((loss[positive].mean()+loss[~positive].mean())/2),
                'max_sampled_context_score_auc': float(roc_auc_score(positive, [r['max_score'] for r in chosen]))}
        result['arms'][name] = {'model_id': model.model_id, 'selected_epoch': model.metadata['training']['selected_epoch'],
                                'summary': summary, 'records': rows}
        print(name, summary, flush=True)
    rows = result['arms']['peak-relative']['records']
    train = [r for r in rows if r['split'] == 'train']
    dev = [r for r in rows if r['split'] == 'development']
    x = np.asarray([[r['mean_input']] for r in train])
    y = np.asarray([r['label'] == 'drone' for r in train])
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(class_weight='balanced', C=1., random_state=42).fit(scaler.transform(x), y)
    p = classifier.predict_proba(scaler.transform([[r['mean_input']] for r in dev]))[:, 1]
    sources = defaultdict(list)
    for row, score in zip(dev, p, strict=True):
        sources[row['source']].append({'id': row['id'], 'label': row['label'], 'score': float(score)})
    result['density_only_control'] = {
        'description': 'A single scalar: mean of all pixels over all sampled contexts in a recording. No frequency positions, harmonic relationships or temporal order. Train-only scaler and balanced logistic regression C1, no selection or threshold tuning.',
        'development_auc': float(roc_auc_score([r['label'] == 'drone' for r in dev], p)),
        'coef_standardized': float(classifier.coef_[0, 0]), 'development_by_source': dict(sources)}
    original = load_model('artifacts/spectral-lab-v1/models/linear256-fft4096', True)
    repeat = load_model(root / 'normalization-models/peak-relative')
    result['baseline_repeat'] = {'original_model_id': original.model_id, 'repeat_model_id': repeat.model_id,
        'maximum_parameter_difference': max(float(np.max(abs(original.arrays[k]-repeat.arrays[k]))) for k in original.arrays),
        'note': 'Same seeded recipe does not reproduce bit for bit across the environment restart/CPU execution. Hard-max selection can amplify small floating-point differences; no claim that the cause is proven. The three new arms run together in the same process.'}
    write_json(report / 'normalization-comparison.json', result)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
