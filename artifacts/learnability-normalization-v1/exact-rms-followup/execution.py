#!/usr/bin/env python3
"""Explicit uncapped numerical-RMS follow-up, motivated by input coverage."""

from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import learnability_normalization_v1 as task
from drone_detector.audio import decode, normalize_context_gain
from drone_detector.common import file_hash, read_json, write_json
from drone_detector.spectrogram import SpectrogramConfig, spectrogram, tensorflow


def main():
    report = Path('reports/learnability-normalization-v1/exact-rms-followup')
    report.mkdir(exist_ok=True)
    root = Path('runs/learnability-normalization-v1')
    config = SpectrogramConfig(axis='linear_filterbank', mel_bins=256, window_seconds=1.,
                               power_reference='unit_stft_power', normalize_rms='exact')
    execution = {'parent_plan_sha256': file_hash('reports/transfer-diagnosis-v1/plan.json'),
                 'input_coverage_audit_sha256': file_hash('reports/transfer-diagnosis-v1/normalization-coverage.json'),
                 'trigger': 'Only3194/5491 clips reached RMS0.05 under20dB/peak caps;40dB still misses70. This follow-up tests an actual shared numerical RMS instead of calling bounded gain equal-RMS.',
                 'features': config.to_dict(), 'epochs': 24, 'seeds': [42, 7],
                 'code_sha256': file_hash(__file__), 'training_helper_sha256': file_hash('scripts/learnability_normalization_v1.py'),
                 'audio_code_sha256': file_hash('src/drone_detector/audio.py'),
                 'spectrogram_code_sha256': file_hash('src/drone_detector/spectrogram.py'),
                 'method': 'Same released labels, weights, splits,154689parameter architecture, initial seeds, batch permutations, optimizer, epochs, validation selection and0.5 threshold. Only amplitude rule differs.',
                 'scope': 'Consumed single-session diagnosis. Numerical normalization may exceed unit float amplitude; no clipping, audio replay normalization or physical SPL calibration. Silence guard retained.'}
    path = report / 'execution.json'
    if path.exists():
        assert read_json(path) == execution
    else:
        write_json(path, execution)
    items = read_json(task.DATA / 'groups.json')['segments']
    assignment = read_json(task.DATA / 'splits/split_interval_disjoint.json')['assignment']
    y = np.array([int(item['label'] == 'drone') for item in items], dtype=np.float32)
    split = {name: task.interval_class_weights(items, assignment, name) for name in ['train', 'val', 'test']}
    cache = root / 'exact-rms-fixed.npy'
    marker = root / 'exact-rms-fixed-cache.json'
    if not marker.exists():
        images = np.lib.format.open_memmap(cache, mode='w+', dtype=np.float32, shape=(len(items), 256, 47))
        errors, above_unit = [], 0
        for index, item in enumerate(items):
            wave, _ = decode(task.DATA / item['file'], 16000)
            gained = normalize_context_gain(wave, config)
            errors.append(abs(float(np.sqrt(np.mean(gained.astype(np.float64) ** 2))) - .05))
            above_unit += int(np.max(np.abs(gained)) > 1)
            images[index] = spectrogram(wave, config)
            if (index + 1) % 1000 == 0:
                print('CACHE exact RMS', index + 1, flush=True)
        images.flush()
        assert max(errors) < 1e-5
        write_json(marker, {'execution_sha256': file_hash(path), 'cache_sha256': file_hash(cache),
                           'rms_target_reached': len(errors), 'max_rms_error': max(errors), 'float_peaks_above_one': above_unit})
    else:
        assert read_json(marker)['execution_sha256'] == file_hash(path)
    images = np.load(cache, mmap_mode='r')
    tf = tensorflow()
    tf.config.experimental.enable_op_determinism()
    task.REPORT = report
    task.ARTIFACT = Path('artifacts/learnability-normalization-v1/exact-rms-followup')
    task.ARTIFACT.mkdir(exist_ok=True)
    control = task.linear_control('exact-rms-fixed', images, items, y, split)
    results = {}
    for seed in [42, 7]:
        results[str(seed)] = task.train_one('exact-rms-fixed', images, config, seed, items, y, split)
        parent = read_json(f'reports/learnability-normalization-v1/peak-seed{seed}.json')
        assert results[str(seed)]['initial_weights_sha256'] == parent['initial_weights_sha256']
    write_json(report / 'comparison.json', {'neural': results, 'non_neural': control, 'input_coverage': read_json(marker)})
    print('COMPLETE exact RMS follow-up', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
