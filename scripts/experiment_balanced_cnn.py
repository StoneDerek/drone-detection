#!/usr/bin/env python3
"""Adaptive, declared control for the observed high-variance minibatch setup."""

import tarfile
from pathlib import Path

from threadpoolctl import threadpool_limits

from drone_detector.common import file_hash, read_json, revision, write_json
from drone_detector.foundation_training import train_foundation_cnn
from drone_detector.spectrogram import SpectrogramConfig, tensorflow


def main():
    root = Path('runs/cnn-foundations-v1')
    report = Path('reports/cnn-foundations-v1')
    output = root / 'trained/medium-balanced-max'
    if (output / 'model.json').exists():
        print('Preserving completed balanced-batch control', flush=True)
        return
    path = report / 'balanced-followup-plan.json'
    if not path.exists():
        write_json(path, {'parent_plan_sha256': file_hash(report / 'plan.json'),
            'code': revision(), 'manifest_sha256': file_hash('runs/real-v2/corpus/manifest-v2.jsonl'),
            'scope': 'Adaptive DEVELOPMENT control, declared after observing the original sampled/max medium training instability. No new assessment or strong labels.',
            'reason': '31% negative-only minibatches; common training loss oscillates widely (.64 to1.53) despite more parameters. Class weights balance expectation, not each batch, and gradient clipping is nonlinear.',
            'intervention': 'Same93313-parameter coverage CNN, cached inputs, seed42, sampled8 hard-max positives, mean negative loss, Adam.001 clipnorm1, dropout.1, L2.0001. Four recordings per class per batch, no inverse-class reweighting (avoids double balancing). Keep51 optimizer steps/epoch and24 epochs.',
            'interpretation': 'This changes class composition variance and repeat frequency while preserving the expected class-balanced objective and optimizer-step budget. It does not isolate clipping alone or add verified audible labels.'})
        with tarfile.open(root / 'balanced-implementation.tar.gz', 'w:gz') as archive:
            for file in [*sorted(Path('src').rglob('*.py')), Path('scripts/experiment_balanced_cnn.py')]:
                archive.add(file, arcname=str(file))
    plan = read_json(path)
    tf = tensorflow()
    tf.config.experimental.enable_op_determinism()
    model = train_foundation_cnn('runs/real-v2/corpus/manifest-v2.jsonl',
        root.parent / 'spectral-lab-v1/cache/linear256-fft4096',
        SpectrogramConfig(axis='linear_filterbank', mel_bins=256), output, plan,
        balanced_batches=True)
    write_json(report / 'medium-balanced-max-training.json', {'model_id': model.model_id,
        'architecture': model.metadata['architecture'],
        'training': {k: v for k, v in model.metadata['training'].items() if k != 'records'}})
    print('Completed balanced control', model.model_id, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
