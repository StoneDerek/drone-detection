#!/usr/bin/env python3
"""A fixed, same-session feasibility task using published human interval labels."""

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from drone_detector.audio import decode
from drone_detector.common import file_hash, read_json, write_json, write_jsonl
from drone_detector.spectrogram import SpectrogramConfig, axes

ROOT = Path('runs/external-v3/downloads/itu/itu-aris-lab-acoustic-drone-dataset')
REPORT = Path('reports/human-label-task-v1')
CONFIG = SpectrogramConfig(axis='linear_filterbank', mel_bins=256)


def one_second_features(audio):
    if len(audio) != 16000:
        raise ValueError('Matched one-second canonical inputs required')
    _, _, stft = signal.stft(audio, fs=16000, nperseg=4096, noverlap=3840,
                            boundary=None, padded=False)
    _, _, bank = axes(CONFIG)
    power = bank @ (np.abs(stft).astype(np.float64) ** 2)
    if power.shape != (256, 47):
        raise ValueError('Unexpected valid-frame dimensions')
    peak = float(power.max())
    image = np.zeros_like(power) if peak <= 1e-30 else np.clip(1 + 10 * np.log10(np.maximum(power / peak, 1e-30)) / 80, 0, 1)
    return np.concatenate([image.mean(axis=1), image.std(axis=1)])


def summarize(y, scores):
    return {'examples': len(y), 'positive': int(y.sum()), 'negative': int((1 - y).sum()),
            'confusion_tn_fp_fn_tp': confusion_matrix(y, scores >= .5, labels=[0, 1]).ravel().tolist(),
            'roc_auc': float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None}


def main():
    REPORT.mkdir(exist_ok=True)
    if (REPORT / 'results.json').exists():
        raise ValueError('A completed fixed task must not be silently rerun/tuned')
    groups = read_json(ROOT / 'groups.json')
    assignment = read_json(ROOT / 'splits/split_interval_disjoint.json')['assignment']
    items = groups['segments']
    bag_splits = defaultdict(set)
    half_fingerprints = defaultdict(set)
    features, labels = [], []
    for index, item in enumerate(items):
        split = assignment[item['file']]
        bag_splits[item['group']].add(split)
        raw, rate = sf.read(ROOT / item['file'], dtype='float32')
        for half in [raw[:rate // 2], raw[rate // 2:]]:
            half_fingerprints[hashlib.sha256(half.tobytes()).hexdigest()].add(split)
        audio, _ = decode(ROOT / item['file'], 16000)
        features.append(one_second_features(audio))
        labels.append(int(item['label'] == 'drone'))
        if (index + 1) % 1000 == 0:
            print('Features', index + 1, '/', len(items), flush=True)
    conflicts = [sorted(splits) for splits in half_fingerprints.values() if len(splits) > 1]
    audit = {'publisher_groups': len(bag_splits), 'groups_crossing_splits': [key for key, value in bag_splits.items() if len(value) != 1],
             'raw_half_second_fingerprints_crossing_splits': len(conflicts),
             'limit': 'Exact half-sample overlap checks do not rule out arbitrary shifted/lossy copies; all data comes from one capture session.'}
    write_json(REPORT / 'split-audit.json', audit)
    if conflicts or audit['groups_crossing_splits']:
        raise ValueError('Published split has overlapping samples or source intervals; resolve before fitting')
    x = np.asarray(features)
    y = np.asarray(labels)
    group_ids = np.array([item['group'] for item in items])
    splits = np.array([assignment[item['file']] for item in items])
    train = splits == 'train'
    train_groups = sorted(set(group_ids[train]))
    weights = np.zeros(len(items))
    for group in train_groups:
        mask = train & (group_ids == group)
        weights[mask] = 1 / mask.sum()
    scaler = StandardScaler().fit(x[train], sample_weight=weights[train])
    z = scaler.transform(x)
    bags = [z[train & (group_ids == group)] for group in train_groups]
    bag_labels = np.array([y[train & (group_ids == group)][0] for group in train_groups])
    classifier = LogisticRegression(C=1, class_weight='balanced', max_iter=2000, random_state=42)
    classifier.fit(np.array([bag.mean(axis=0) for bag in bags]), bag_labels)
    for _ in range(4):
        selected, targets, sample_weights = [], [], []
        for bag, label in zip(bags, bag_labels, strict=True):
            chosen = bag[np.argmax(classifier.decision_function(bag)):][:1] if label else bag
            selected.extend(chosen)
            targets.extend([label] * len(chosen))
            sample_weights.extend([1 / len(chosen)] * len(chosen))
        targets, sample_weights = np.array(targets), np.array(sample_weights)
        for label in [0, 1]:
            sample_weights[targets == label] *= len(targets) / (2 * sample_weights[targets == label].sum())
        classifier = LogisticRegression(C=1, max_iter=2000, random_state=42)
        classifier.fit(np.asarray(selected), targets, sample_weight=sample_weights)
        if classifier.n_iter_[0] >= 2000:
            raise ValueError('Fixed classifier did not converge')
    scores = classifier.predict_proba(z)[:, 1]
    result = {'scope': 'Within-one-session feasibility under published labels, not independent-session field validation.',
              'label_limit': 'Drone intervals were manually selected by2 annotators (reported kappa0.5869); background came from unlabelled stretches. Audible onset/offset/absence truth remains unverified.',
              'protocol_sha256': file_hash('reports/external-v3/compatibility-amendment.json'),
              'execution_sha256': file_hash(__file__), 'threshold': .5, 'seed': 42, 'C': 1,
              'scaler_fit_split': 'train', 'training_groups': train_groups, 'splits': {}}
    for split in ['train', 'val', 'test']:
        mask = splits == split
        bag_scores, bag_truth = [], []
        for group in sorted(set(group_ids[mask])):
            group_mask = mask & (group_ids == group)
            bag_scores.append(float(scores[group_mask].max()))
            bag_truth.append(int(y[group_mask][0]))
        result['splits'][split] = {'segments': summarize(y[mask], scores[mask]),
                                  'publisher_interval_blocks': summarize(np.array(bag_truth), np.array(bag_scores))}
    destination = Path('artifacts/human-label-task-v1')
    destination.mkdir(exist_ok=True)
    np.savez_compressed(destination / 'parameters.npz', mean=scaler.mean_, scale=scaler.scale_, coef=classifier.coef_[0], intercept=classifier.intercept_)
    write_json(destination / 'model.json', {'family': '512descriptor MI logistic', 'input': '1s,256uniform-Hz logpower,47 valid frames,mean/std',
                'features': CONFIG.to_dict(), 'decision_threshold': .5, 'parameter_sha256': file_hash(destination / 'parameters.npz'),
                'scope': result['scope'], 'protocol_sha256': result['protocol_sha256']})
    write_jsonl(REPORT / 'predictions.jsonl', [{**item, 'split': split, 'score': float(score)} for item, split, score in zip(items, splits, scores, strict=True)])
    write_json(REPORT / 'results.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
