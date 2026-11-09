#!/usr/bin/env python3
"""Declared causal follow-ups to the CNN training/representation controls."""

import argparse
import tarfile
from pathlib import Path

from threadpoolctl import threadpool_limits

from drone_detector.common import file_hash, read_json, revision, write_json
from drone_detector.foundation_training import train_foundation_cnn
from drone_detector.spectrogram import SpectrogramConfig, tensorflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['full-max', 'heads'])
    args = parser.parse_args()
    root, report = Path('runs/cnn-foundations-v1'), Path('reports/cnn-foundations-v1')
    plan_path = report / f'{args.kind}-followup-plan.json'
    if not plan_path.exists():
        write_json(plan_path, {'code': revision(), 'manifest_sha256': file_hash('runs/real-v2/corpus/manifest-v2.jsonl'),
            'parent_plan_sha256': file_hash(report / 'plan.json'),
            'scope': 'Additional DEVELOPMENT controls; same cached real inputs and weak labels, no new final assessment or multi-view input.',
            'kind': args.kind,
            'full_max_trigger': 'Prespecified decomposition condition met at full/smooth epoch12: common development bag loss.58169 versus sampled/max best24.64274, improvement>.05.',
            'full_max_control': 'Same93313-parameter network, original random recording batches/class weights, all positive contexts and negative sample8. Change only positive pooling from normalized logsumexp to hard max.24epochs, seed42.',
            'head_hypothesis': 'After fixing coverage, global averaging still destroys absolute frequency arrangement. Preserve frequency positions in the readout without adding an input view.',
            'head_arms': {'medium-balanced-global-head': {'variant': 'global_readout', 'parameters': 154675},
                          'medium-balanced-frequency-head': {'variant': 'frequency_aware', 'parameters': 154689}},
            'head_control': 'Same32/64/128 convolution encoder, strides2/3/3, seed42, cached256x172 inputs, balanced4+4 recording batches, sampled8/max positive and mean-negative loss, optimizer and24epochs. Global pooling plus Dense473 is parameter-matched within14 parameters to time-only pooling plus Dense32 over15 frequency positions. Nonlinear head shape and pooling differ intentionally; parameter count alone is controlled, not identical function classes.'})
        with tarfile.open(root / f'{args.kind}-implementation.tar.gz', 'w:gz') as archive:
            for p in [*sorted(Path('src').rglob('*.py')), Path('scripts/experiment_cnn_followup.py')]:
                archive.add(p, arcname=str(p))
    plan = read_json(plan_path)
    tf = tensorflow()
    tf.config.experimental.enable_op_determinism()
    arms = [('medium-full-max', 'coverage_medium')] if args.kind == 'full-max' else [
        ('medium-balanced-global-head', 'global_readout'), ('medium-balanced-frequency-head', 'frequency_aware')]
    for name, variant in arms:
        output = root / 'trained' / name
        if (output / 'model.json').exists():
            print('Preserving completed follow-up', name, flush=True)
            continue
        model = train_foundation_cnn('runs/real-v2/corpus/manifest-v2.jsonl',
            'runs/spectral-lab-v1/cache/linear256-fft4096', SpectrogramConfig(axis='linear_filterbank', mel_bins=256),
            output, plan, variant=variant, full_positive=args.kind == 'full-max', balanced_batches=args.kind == 'heads')
        write_json(report / f'{name}-training.json', {'model_id': model.model_id,
            'architecture': model.metadata['architecture'],
            'training': {k: v for k, v in model.metadata['training'].items() if k != 'records'}})
        print('Completed follow-up', name, model.model_id, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
