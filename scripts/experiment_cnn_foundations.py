#!/usr/bin/env python3
"""Controlled CNN coverage/capacity/training-formulation investigation."""

import argparse
import tarfile
from pathlib import Path

from threadpoolctl import threadpool_limits

from drone_detector.common import file_hash, hardware, read_json, revision, write_json
from drone_detector.spectral_training import train_cnn
from drone_detector.spectrogram import SpectrogramConfig, tensorflow


ROOT = Path('runs/cnn-foundations-v1')
REPORT = Path('reports/cnn-foundations-v1')
MANIFEST = Path('runs/real-v2/corpus/manifest-v2.jsonl')
CACHE = Path('runs/spectral-lab-v1/cache/linear256-fft4096')


def plan():
    REPORT.mkdir(parents=True, exist_ok=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    path = REPORT / 'plan.json'
    if not path.exists():
        write_json(path, {'id': 'cnn-foundations-v1', 'code': revision(), 'hardware': hardware(),
            'manifest_sha256': file_hash(MANIFEST), 'cache_index_sha256': file_hash(CACHE / 'index.json'),
            'scope': 'TRAIN/DEVELOPMENT only. Existing weak publisher labels; no new independent assessment, invented audible intervals or default promotion. No multi-view/3D input.',
            'input': 'Same verified cached linear256 by172 relative-log-power tensors, 4096 Hann, hop256, 3s context. No amplitude or input representation change.',
            'arms': {
                'compact8': {'variant': 'compact', 'epochs': 8, 'positive': 'sample8/max', 'parameters': 1781},
                'coverage8': {'variant': 'coverage', 'epochs': 8, 'positive': 'sample8/max', 'parameters': 1781},
                'medium-sampled-max': {'variant': 'coverage_medium', 'epochs': 24, 'snapshot_epochs': [8], 'positive': 'sample8/max', 'parameters': 93313},
                'medium-full-smooth': {'variant': 'coverage_medium', 'epochs': 24, 'snapshot_epochs': [8], 'positive': 'all cached contexts, normalized logsumexp temperature0.5', 'parameters': 93313}},
            'controls': 'Seed42, Adam.001 clipnorm1, dropout.1, L2.0001, eight recording bags per optimizer step, balanced/equal recording mass. Negatives sample8 and retain mean window softplus. Same uniformly capped60 candidate contexts/file.',
            'coverage': 'Change frequency strides4 to3 in conv2/conv3, retaining all kernel shapes, channels, parameter count and seeded initial weights. This also changes output stride and receptive-field span; field changes cannot isolate missing rows alone.',
            'selection': 'Common max-positive / mean-negative balanced development bag loss every2 epochs. Compare at8 epochs as well as completed24. Report inference-mode full sampled-training loss, per-source outcomes and gradient/instance-selection exposure.',
            'formulation': 'Full-positive/smooth arm jointly removes random positive sub-bag omission and concentrates less gradient on one winner. It does not repair wrong whole-recording labels. This is a grouped intervention, not proof either component alone causes improvement.',
            'conditional_followup': 'If full/smooth beats sampled/max at24 by>.03 development AUC or>.05 common bag loss, run a full-positive/hard-max arm to separate the two training changes. Repeat a clearly improved result at seed7 before interpreting stability.',
            'calibration': 'Any later interface alarm calibration uses the same disclosed5% development-negative budget, with source-level failures and raw ranking retained. No new field-FPR claim.'})
        with tarfile.open(ROOT / 'initial-source.tar.gz', 'w:gz') as archive:
            for item in [*sorted(Path('src').rglob('*.py')), Path(__file__), Path('pyproject.toml'), Path('uv.lock')]:
                archive.add(item, arcname=str(item))
    return read_json(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['controls', 'training'])
    args = parser.parse_args()
    experiment = plan()
    tf = tensorflow()
    tf.config.experimental.enable_op_determinism()
    config = SpectrogramConfig(axis='linear_filterbank', mel_bins=256)
    for name in (['compact8', 'coverage8'] if args.stage == 'controls' else ['medium-sampled-max', 'medium-full-smooth']):
        output = ROOT / 'trained' / name
        if (output / 'model.json').exists():
            print('Preserving completed arm', name, flush=True)
            continue
        arm = experiment['arms'][name]
        if args.stage == 'controls':
            model = train_cnn(MANIFEST, CACHE, config, output, experiment,
                              variant=arm['variant'], epochs=arm['epochs'])
        else:
            from drone_detector.foundation_training import train_foundation_cnn
            model = train_foundation_cnn(MANIFEST, CACHE, config, output, experiment,
                full_positive=name.endswith('full-smooth'), smooth_positive=name.endswith('full-smooth'))
        write_json(REPORT / f'{name}-training.json', {'model_id': model.model_id,
            'architecture': model.metadata['architecture'],
            'training': {k: v for k, v in model.metadata['training'].items() if k != 'records'}})
        print('Completed arm', name, model.model_id, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
