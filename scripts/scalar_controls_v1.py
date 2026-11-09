#!/usr/bin/env python3
"""Train-fitted volume/brightness controls for the consumed simplified task."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import learnability_normalization_v1 as task
from drone_detector.common import read_json, write_json


def main():
    items = read_json(task.DATA / 'groups.json')['segments']
    assignment = read_json(task.DATA / 'splits/split_interval_disjoint.json')['assignment']
    y = np.array([int(row['label'] == 'drone') for row in items])
    splits = {name: task.interval_class_weights(items, assignment, name) for name in ['train', 'val', 'test']}
    train, weights = splits['train']
    audit = read_json('reports/transfer-diagnosis-v1/normalization-coverage.json')
    by_file = {row['file']: row for row in audit['records']}
    features = {'log_waveform_rms': np.array([np.log(max(by_file[row['file']]['raw_rms'], 1e-12)) for row in items])}
    for name in ['peak', 'fixed', 'rms-fixed']:
        images = np.load(task.ROOT / f'{name}.npy', mmap_mode='r')
        features[name + '_image_brightness'] = images.mean(axis=(1, 2)).astype(float)
    results = {}
    for name, values in features.items():
        scaler = StandardScaler().fit(values[train, None], sample_weight=weights[train])
        z = scaler.transform(values[:, None])
        model = LogisticRegression(C=1, max_iter=2000, random_state=42).fit(z[train], y[train], sample_weight=weights[train])
        scores = model.predict_proba(z)[:, 1]
        results[name] = {'splits': {key: task.metrics(items, ids, y, scores) for key, (ids, _) in splits.items()},
                         'train_mean': scaler.mean_.tolist(), 'train_scale': scaler.scale_.tolist(),
                         'coefficient': model.coef_.tolist(), 'intercept': model.intercept_.tolist()}
    write_json('reports/transfer-diagnosis-v1/scalar-controls.json', results)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
