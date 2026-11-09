#!/usr/bin/env python3
"""Repair the demonstrated noise loophole, then compare matched readouts."""

import argparse
import tarfile
from pathlib import Path

from threadpoolctl import threadpool_limits

from drone_detector.common import file_hash, read_json, revision, write_json
from drone_detector.foundation_training import train_foundation_cnn
from drone_detector.spectrogram import SpectrogramConfig, tensorflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('readout', choices=['global', 'frequency'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    root, report = Path('runs/cnn-foundations-v1'), Path('reports/cnn-foundations-v1')
    plan_path = report / 'coherent-followup-plan.json'
    if not plan_path.exists():
        write_json(plan_path, {
            'code': revision(), 'manifest_sha256': file_hash('runs/real-v2/corpus/manifest-v2.jsonl'),
            'parent_plan_sha256': file_hash(report / 'plan.json'),
            'trigger': 'Exact input-independent dropout counterexample, corroborated by fixed-weight real-input checkpoint audit.',
            'diagnosis_sha256': file_hash(report / 'mil-objective-audit.json'),
            'repair': 'Binary cross entropy of the SAME maximum window logit for either recording class. Disable dropout before instance pooling. These two repairs are bundled; do not attribute any gain solely to one.',
            'sampling': 'Same sampled8 cyclic policy for both classes, same four recordings per class/batch,51 steps/epoch. No new data or labels.',
            'input': 'Frozen256x172 uniform-Hz relative log-power frontend, unchanged7299 cached train/development contexts.',
            'arms': {'global': {'variant': 'global_readout', 'parameters': 154675},
                     'frequency': {'variant': 'frequency_aware', 'parameters': 154689}},
            'control': 'Same32/64/128 encoder, strides2/3/3, seed42, Adam.001 clipnorm1 L2.0001. Only readout differs between paired repair arms.',
            'epochs': 24, 'snapshot': 8,
            'selection': 'Minimum deterministic consistent-max balanced DEVELOPMENT loss every2 epochs; also retain last and best-at8.',
            'replication': 'Seed7 on the stronger repaired arm if development max-score AUC improves>.03 over original linear256 baseline or catches >=3 more positive development recordings under the same5% negative budget.',
            'scope': 'Development research, weak recording labels and only6 conservative positive training groups. No final-assessment audio or independent field reliability claim.'})
        with tarfile.open(root / 'coherent-implementation.tar.gz', 'w:gz') as archive:
            for path in [*sorted(Path('src').rglob('*.py')), Path('scripts/experiment_coherent_cnn.py')]:
                archive.add(path, arcname=str(path))
    plan = read_json(plan_path)
    name = f'coherent-{args.readout}-seed{args.seed}'
    output = root / 'trained' / name
    if (output / 'model.json').exists():
        raise ValueError('Preserving existing completed run')
    tf = tensorflow()
    tf.config.experimental.enable_op_determinism()
    model = train_foundation_cnn('runs/real-v2/corpus/manifest-v2.jsonl',
        'runs/spectral-lab-v1/cache/linear256-fft4096', SpectrogramConfig(axis='linear_filterbank', mel_bins=256),
        output, plan, seed=args.seed, variant=plan['arms'][args.readout]['variant'],
        balanced_batches=True, consistent_loss=True, dropout_rate=0.)
    write_json(report / f'{name}-training.json', {'model_id': model.model_id,
        'architecture': model.metadata['architecture'],
        'training': {k: v for k, v in model.metadata['training'].items() if k != 'records'}})
    print('Completed repaired run', name, model.model_id, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
