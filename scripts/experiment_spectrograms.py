#!/usr/bin/env python3
"""Real-audio development experiments; preserves the assessed default."""

import argparse
import tarfile
from dataclasses import asdict
from pathlib import Path

from threadpoolctl import threadpool_limits

from drone_detector.common import file_hash, read_json, read_jsonl, revision, write_json
from drone_detector.model import load_model
from drone_detector.spectral_training import prepare_cache, train_cnn, train_svm
from drone_detector.spectrogram import SpectrogramConfig, axes
from drone_detector.weak import tune_weak

CASES = {
    'mel64-fft1024': SpectrogramConfig(axis='mel', n_fft=1024),
    'mel64-fft4096': SpectrogramConfig(axis='mel', n_fft=4096),
    'mel256-fft4096': SpectrogramConfig(axis='mel', n_fft=4096, mel_bins=256),
    'linear-fft1024': SpectrogramConfig(n_fft=1024),
    'linear-fft4096': SpectrogramConfig(n_fft=4096),
    'linear-power-fft4096': SpectrogramConfig(n_fft=4096, intensity='linear_power'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='runs/spectral-lab-v1')
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    parser.add_argument('--epochs', type=int, default=8)
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = Path('runs/real-v2/corpus/manifest-v2.jsonl')
    rows = read_jsonl(manifest)
    plan_path = root / 'plan.json'
    if not plan_path.exists():
        plan = {'code': revision(), 'manifest_sha256': file_hash(manifest),
            'scope': 'DEVELOPMENT ONLY; prior assessment consumed; no fresh final set scored; assessed default preserved',
            'hypotheses': ['Retaining linear frequency bins preserves narrow harmonic detail',
                           'Increasing actual Hann analysis window from 1024 to 4096 improves frequency discrimination at the cost of temporal resolution',
                           'Log intensity may expose weak harmonics better than linear power',
                           'Identical CNN capacity can respond differently to frequency/time tensor dimensions',
                           'A fully non-neural RBF SVM on spectral temporal summaries can compete with the CNN'],
            'controls': 'Same grouped real files, uniformly sampled contexts, 16kHz mono, 3s context, .5s detection step, 256-sample STFT hop, 50-7600Hz band, per-context relative amplitude, CNN architecture/optimizer/seed/epoch budget and weak bag objective',
            'interpretation_limit': 'Dense grids change effective CNN receptive fields in physical Hz; higher input pixel count is not independent data or proof of increased information. Log intensity and mel frequency warping are separate factors.',
            'cases': {name: {'features': asdict(config), 'shape': [len(axes(config)[0]), len(axes(config)[1]), 1]} for name, config in CASES.items()},
            'seed': 42, 'epochs': args.epochs,
            'data': {split: {'files': len(part := [r for r in rows if r['split'] == split]),
                             'positive': sum(r['label'] == 'drone' for r in part),
                             'negative': sum(r['label'] == 'non_drone' for r in part),
                             'hours': sum(r['duration_seconds'] for r in part) / 3600}
                     for split in ['train', 'development']},
            'checkpoint_selection': 'Minimum balanced development bag loss on <=60 uniform contexts per recording, checked every 2 epochs; full development recording traces used once for decision calibration',
            'source_video': {'url': 'https://www.youtube.com/watch?v=UytYrhYLOKI', 'inspection': 'YouTube oEmbed request blocked with proxy 403; content not watched or verified'},
            'attribution': 'Actual gradient-weighted activations of trained CNN logits. Not proof of causal acoustic cues; no Grad-CAM claimed for SVM or frozen SavedModel.',
            'promotion': 'No default replacement or fresh reliability claim from this development sweep'}
        write_json(plan_path, plan)
        with tarfile.open(root / 'training-source.tar.gz', 'w:gz') as archive:
            for file in sorted(Path('src').rglob('*.py')):
                archive.add(file, arcname=str(file))
            for name in ['scripts/experiment_spectrograms.py', 'pyproject.toml', 'uv.lock']:
                archive.add(name, arcname=name)
    plan = read_json(plan_path)
    if plan['manifest_sha256'] != file_hash(manifest) or plan['epochs'] != args.epochs:
        raise ValueError('Cannot change the frozen development plan in place')
    implementation = revision()
    revisions = root / 'source-revisions'
    revisions.mkdir(exist_ok=True)
    snapshot = revisions / (implementation['source_sha256'] + '.tar.gz')
    if not snapshot.exists():
        with tarfile.open(snapshot, 'w:gz') as archive:
            for file in sorted(Path('src').rglob('*.py')):
                archive.add(file, arcname=str(file))
            archive.add('scripts/experiment_spectrograms.py')
        write_json(snapshot.with_suffix('.json'), {'code': implementation,
            'note': 'Equivalent sparse mel projection replaces dense zero multiplications; original completed arms retained. Hypotheses, data, tensors, training objective and selection rules unchanged.'})
    plan = {**plan, 'implementation_code': implementation}
    summary = read_json(root / 'results.json') if (root / 'results.json').exists() else {'scope': plan['scope'], 'models': {}}
    with threadpool_limits(limits=1):
        for name in args.cases:
            config = CASES[name]
            print(f'Starting planned arm {name}: {plan["cases"][name]["shape"]}', flush=True)
            cache = root / 'cache' / name
            trained, frozen = root / 'trained' / name, root / 'models' / name
            if not trained.exists():
                prepare_cache(manifest, config, cache)
                train_cnn(manifest, cache, config, trained, plan, seed=42, epochs=args.epochs)
            if not frozen.exists():
                tune_weak(trained, manifest, frozen, policy='sensitivity-v2')
            model = load_model(frozen, True)
            summary['models'][name] = {'family': model.metadata['family'], 'model_id': model.model_id,
                'features': model.metadata['features'], 'shape': plan['cases'][name]['shape'],
                'development': model.metadata['selection']['chosen'],
                'training': {k: v for k, v in model.metadata['training'].items() if k != 'records'}}
            write_json(root / 'results.json', summary)
            print(f'Finished {name}: {summary["models"][name]["development"]}', flush=True)
        config = CASES['linear-fft4096']
        cache = root / 'cache' / 'linear-fft4096'
        prepare_cache(manifest, config, cache)
        name = 'svm-linear-fft4096'
        trained, frozen = root / 'trained' / name, root / 'models' / name
        if not trained.exists():
            train_svm(cache, config, trained, plan)
        if not frozen.exists():
            tune_weak(trained, manifest, frozen, policy='sensitivity-v2')
        model = load_model(frozen, True)
        summary['models'][name] = {'family': model.metadata['family'], 'model_id': model.model_id,
            'features': model.metadata['features'], 'development': model.metadata['selection']['chosen'],
            'training': {k: v for k, v in model.metadata['training'].items() if k != 'records'}}
        write_json(root / 'results.json', summary)
        print(f'Finished non-neural arm: {summary["models"][name]["development"]}', flush=True)


if __name__ == '__main__':
    main()
