#!/usr/bin/env python3
"""Three controlled DEVELOPMENT arms, not a model search or promotion."""

from dataclasses import replace
from pathlib import Path

from threadpoolctl import threadpool_limits

from drone_detector.common import file_hash, read_json, revision, write_json
from drone_detector.spectral_training import prepare_cache, train_cnn
from drone_detector.spectrogram import SpectrogramConfig


def main():
    root = Path('runs/cnn-diagnosis-v1')
    report = Path('reports/cnn-diagnosis-v1')
    manifest = Path('runs/real-v2/corpus/manifest-v2.jsonl')
    original_cache = Path('runs/spectral-lab-v1/cache/linear256-fft4096')
    config = SpectrogramConfig(axis='linear_filterbank', mel_bins=256)
    arms = {'peak-relative': config,
            'fixed-reference': replace(config, power_reference='unit_stft_power'),
            'rms-fixed-reference': replace(config, power_reference='unit_stft_power', normalize_rms=True)}
    plan_path = report / 'normalization-plan.json'
    if not plan_path.exists():
        write_json(plan_path, {'code': revision(), 'manifest_sha256': file_hash(manifest),
            'scope': 'Controlled DEVELOPMENT diagnosis; no final audio, threshold optimization, default change or multi-view input.',
            'hypothesis': 'A later peak normalization cancels upstream scalar RMS gain. A fixed STFT power reference permits reference-RMS changes to reach the CNN.',
            'arms': {name: cfg.to_dict() for name, cfg in arms.items()},
            'controls': 'Same 405 train/210 development files, starts/max60 contexts, compact1781 CNN, seed42, eight epochs, sampled-bag objective, optimizer and checkpoint selection. Baseline is repeated to measure run reproducibility.',
            'amplitude': 'Reference1 means scipy spectrum-scaled STFT power, not calibrated sound pressure. RMS target.05, gain ceiling20dB, peak ceiling.98, available context only. Caps mean RMS target is not always reached.',
            'interpretation': 'Changing representation changes the optimization landscape. One seed and weak labels cannot identify a universal architecture winner. Raw max60-context development bag loss is the prespecified comparison; no field reliability claim.'})
    plan = read_json(plan_path)
    for name, cfg in arms.items():
        output = root / 'normalization-models' / name
        if output.exists():
            print('Preserving completed arm', name, flush=True)
            continue
        if name == 'peak-relative':
            cache = original_cache
        else:
            cache = root / 'cache' / name
            prepare_cache(manifest, cfg, cache)
        model = train_cnn(manifest, cache, cfg, output, plan)
        write_json(report / f'normalization-{name}-training.json', {
            'model_id': model.model_id, 'features': model.metadata['features'],
            'training': {k: v for k, v in model.metadata['training'].items() if k != 'records'},
            'status': 'trained diagnostic, not frozen or deployed'})
        print('Completed normalization arm', name, model.model_id, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
