#!/usr/bin/env python3
"""Portable real-input checks for all paired diagnostic packages."""

from pathlib import Path

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from drone_detector.audio import decode
from drone_detector.common import file_hash, read_json, write_json
from drone_detector.model import load_model
from drone_detector.spectrogram import SpectrogramConfig, spectrogram


def main():
    data = Path('runs/external-v3/downloads/itu/itu-aris-lab-acoustic-drone-dataset')
    root = Path('reports/learnability-normalization-v1')
    artifacts = Path('artifacts/learnability-normalization-v1')
    cache = Path('runs/learnability-normalization-v1')
    items = read_json(data / 'groups.json')['segments']
    assignment = read_json(data / 'splits/split_interval_disjoint.json')['assignment']
    selected = [next(i for i, row in enumerate(items) if assignment[row['file']] == split and row['label'] == label)
                for split in ['train', 'val', 'test'] for label in ['drone', 'background']]
    configs = read_json(root / 'execution.json')['arms']
    configs['exact-rms-fixed'] = read_json(root / 'exact-rms-followup/execution.json')['features']
    checks = []
    for name, configuration in configs.items():
        parent = artifacts if name != 'exact-rms-fixed' else artifacts / 'exact-rms-followup'
        for seed in [42, 7]:
            model = load_model(parent / f'{name}-seed{seed}')
            expected = np.load(cache / f'{name}-seed{seed}-scores.npy', allow_pickle=False)
            for index in selected:
                wave, _ = decode(data / items[index]['file'], 16000)
                score = model.score(wave)
                error = abs(score - float(expected[index]))
                assert error < 1e-5
                checks.append({'model': f'{name}-seed{seed}', 'file': items[index]['file'], 'error': error})
        destination = parent / f'{name}-logistic'
        parameters = np.load(destination / 'parameters.npz', allow_pickle=False)
        expected = np.load(cache / f'{name}-logistic-scores.npy', allow_pickle=False)
        cfg = SpectrogramConfig(**configuration)
        for index in selected:
            wave, _ = decode(data / items[index]['file'], 16000)
            image = spectrogram(wave, cfg)
            features = np.concatenate([image.mean(axis=1), image.std(axis=1)]).astype(np.float64)
            score = float(expit(((features - parameters['mean']) / parameters['scale']) @ parameters['coef'] + parameters['intercept'][0]))
            error = abs(score - float(expected[index]))
            assert error < 1e-8
            checks.append({'model': f'{name}-logistic', 'file': items[index]['file'], 'error': error})
        # Make the standalone descriptor package explicit about its full input,
        # rather than requiring lookup of the arm name in the parent execution.
        metadata = read_json(destination / 'model.json')
        metadata['features'] = configuration
        metadata['normalization_and_weights'] = 'Fit train only; released-label diagnostic. Numeric parameters unchanged.'
        write_json(destination / 'model.json', metadata)
    write_json(root / 'portable-verification.json', {'checks': checks, 'passed': True,
               'maximum_error': max(row['error'] for row in checks), 'code_sha256': file_hash(__file__),
               'scope': 'Six prespecified real inputs per numeric package; software reproducibility, not new acoustic trials.'})
    print('Verified', len(checks), 'real-input package scores')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
