#!/usr/bin/env python3
"""Falsifiable controls before changing the detector or training recipe.

Only existing TRAIN/DEVELOPMENT audio is opened. Artificial image pairs below
are architecture unit controls, never evidence of acoustic detection accuracy.
"""

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import signal
from threadpoolctl import threadpool_limits

from drone_detector.audio import FeatureConfig, normalize_context_gain
from drone_detector.common import file_hash, read_json, write_json
from drone_detector.data import checked_audio, load_manifest
from drone_detector.model import load_model
from drone_detector.spectrogram import SpectrogramConfig, axes, build_cnn, spectrogram, tensorflow


ROOT = Path('runs/cnn-diagnosis-v1')
REPORT = Path('reports/cnn-diagnosis-v1')
MANIFEST = Path('runs/real-v2/corpus/manifest-v2.jsonl')
CACHE = Path('runs/spectral-lab-v1/cache/linear256-fft4096')


def fixed_reference_image(wave, config):
    """STFT power relative to ONE fixed value (1), rather than each clip's max.

    scipy's spectrum-scaled STFT is retained; 1 denotes unit STFT magnitude,
    not a calibrated sound pressure or a measurement of acoustic SNR.
    """
    wave = np.pad(wave, (0, config.window_samples - len(wave)))
    _, _, values = signal.stft(wave, fs=config.sample_rate, nperseg=config.n_fft,
        noverlap=config.n_fft - config.stft_hop, boundary=None, padded=False)
    power = np.abs(values).astype(np.float64) ** 2
    _, _, bank = axes(config)
    power = bank @ power
    return np.clip(1 + 10 * np.log10(np.maximum(power, 1e-30)) / config.dynamic_range_db,
                   0, 1).astype(np.float32)


def audit_inputs():
    started = time.perf_counter()
    index = read_json(CACHE / 'index.json')
    config = SpectrogramConfig(**index['identity']['features'])
    records = {r['id']: r for r in load_manifest(MANIFEST, True)
               if r['split'] in {'train', 'development'}}
    assert file_hash(MANIFEST) == index['identity']['manifest_sha256']
    groups = defaultdict(Counter)
    for row in records.values():
        groups[row['source']][row['label']] += 1
    source_majority = {}
    for source in groups:
        counts = Counter(r['label'] for r in records.values()
                         if r['source'] == source and r['split'] == 'train')
        source_majority[source] = counts.most_common(1)[0][0] if counts else 'non_drone'
    summary = {'scope': 'TRAIN/DEVELOPMENT only; no new independent assessment',
        'manifest_sha256': file_hash(MANIFEST), 'script_sha256': file_hash(__file__),
        'cached_files': len(index['records']), 'cached_contexts_checked': 0,
        'max_cache_image_error': 0., 'records': [], 'source_label_counts': dict(groups),
        'source_majority_control': {}}
    for split in ['train', 'development']:
        selected = [r for r in records.values() if r['split'] == split]
        tp = sum(r['label'] == 'drone' and source_majority[r['source']] == 'drone' for r in selected)
        fp = sum(r['label'] == 'non_drone' and source_majority[r['source']] == 'drone' for r in selected)
        summary['source_majority_control'][split] = {'true_positive': tp, 'false_positive': fp,
            'positives': sum(r['label'] == 'drone' for r in selected),
            'negatives': sum(r['label'] == 'non_drone' for r in selected),
            'correct': sum(r['label'] == source_majority[r['source']] for r in selected)}
    gain_config = FeatureConfig(representation='yamnet_rms')
    model = load_model('artifacts/spectral-lab-v1/models/linear256-fft4096', True)
    for number, cached in enumerate(index['records']):
        row = records[cached['id']]
        for key in ['source', 'group_id', 'sha256', 'split', 'label', 'origin']:
            assert row[key] == cached[key], (row['id'], key)
        assert file_hash(CACHE / cached['cache']) == cached['cache_sha256']
        audio = checked_audio(row, MANIFEST)
        images = np.load(CACHE / cached['cache'], mmap_mode='r', allow_pickle=False)
        assert len(images) == len(cached['starts']) == cached['sampled_windows']
        for start, old in zip(cached['starts'], images, strict=True):
            wave = audio[start:start + config.window_samples]
            fresh = spectrogram(wave, config)
            error = float(np.max(np.abs(fresh - old)))
            summary['max_cache_image_error'] = max(summary['max_cache_image_error'], error)
            summary['cached_contexts_checked'] += 1
        start = cached['starts'][len(cached['starts']) // 2]
        wave = audio[start:start + config.window_samples]
        gained = normalize_context_gain(wave, gain_config)
        relative, relative_gain = spectrogram(wave, config), spectrogram(gained, config)
        fixed, fixed_gain = fixed_reference_image(wave, config), fixed_reference_image(gained, config)
        rms = lambda x: float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))  # noqa: E731
        before, after = rms(wave), rms(gained)
        score = model.score_images(np.stack([relative, relative_gain]))
        summary['records'].append({'id': row['id'], 'source': row['source'], 'label': row['label'],
            'split': row['split'], 'start_sample': start, 'rms_before': before, 'rms_after': after,
            'gain': after / before, 'target_reached': abs(after - .05) < 1e-6,
            'peak_relative_max_error': float(np.max(np.abs(relative - relative_gain))),
            'peak_relative_mean_error': float(np.mean(np.abs(relative - relative_gain))),
            'peak_relative_score_error': float(abs(score[0] - score[1])),
            'fixed_reference_mean_change': float(np.mean(np.abs(fixed - fixed_gain)))})
        if (number + 1) % 40 == 0:
            print(f"Input audit {number + 1}/{len(index['records'])}, {summary['cached_contexts_checked']} contexts", flush=True)
    summary['seconds'] = time.perf_counter() - started
    summary['normalization_summary'] = {
        'one_middle_sampled_context_per_file': len(summary['records']),
        'rms_target_reached': sum(r['target_reached'] for r in summary['records']),
        'max_relative_image_error': max(r['peak_relative_max_error'] for r in summary['records']),
        'max_relative_score_error': max(r['peak_relative_score_error'] for r in summary['records']),
        'median_fixed_reference_mean_change': float(np.median([r['fixed_reference_mean_change'] for r in summary['records']]))}
    write_json(REPORT / 'input-audit.json', summary)


def architecture_control():
    tf = tensorflow()
    # Five narrow bands; identical local motifs and stride phase, but different
    # frequency relationships. Keep every band away from padded boundaries.
    harmonic = [64, 128, 192, 256, 320]
    irregular = [64, 160, 288, 448, 640]
    images = np.zeros((2, 1933, 172, 1), dtype=np.float32)
    for image, bins in zip(images, [harmonic, irregular], strict=True):
        image[np.asarray(bins) - 13, :, 0] = .8
    results = []
    for name in ['linear-fft4096', 'frequency-context-fft4096']:
        model = load_model(Path('artifacts/spectral-lab-v1/models') / name, True)
        net = model.network()
        pooled = tf.keras.Model(net.inputs, net.get_layer('pool').output)(images).numpy()
        results.append({'model': name, 'model_id': model.model_id,
            'frequency_receptive_field_bins': model.metadata['architecture'].get('frequency_receptive_field_bins', 25),
            'score_pair': model.score_images(images[..., 0]).astype(float).tolist(),
            'max_pooled_feature_difference': float(np.max(abs(pooled[0] - pooled[1])))})
    net = build_cnn(42)
    pooled = tf.keras.Model(net.inputs, net.get_layer('pool').output)(images).numpy()
    results.append({'model': 'untrained compact seed42',
        'max_pooled_feature_difference': float(np.max(abs(pooled[0] - pooled[1])))})
    result = {'scope': 'Artificial image architecture control, NOT acoustic performance evidence',
        'harmonic_fft_bins': harmonic, 'irregular_fft_bins': irregular,
        'fft_bin_hz': 16000 / 4096, 'results': results,
        'interpretation_limit': 'An existence proof of a compact CNN invariance, not evidence these exact inputs occur in field audio or that this alone explains the source gap.'}
    write_json(REPORT / 'architecture-control.json', result)
    print('Architecture control:', result, flush=True)


def memorization_control():
    tf = tensorflow()
    index = read_json(CACHE / 'index.json')
    source = sorted([r for r in index['records'] if r['split'] == 'train' and r['source'] == 'svanstrom'], key=lambda r: r['id'])
    selected = []
    for prefix, count in [('DRONE', 8), ('HELICOPTER', 4), ('BACKGROUND', 4)]:
        choices = [r for r in source if prefix in r['id'].upper()]
        if len(choices) < count:
            raise ValueError(f'Insufficient {prefix} records: {len(choices)}')
        selected.extend(choices[:count])
    x = np.stack([np.load(CACHE / r['cache'], allow_pickle=False)[len(r['starts']) // 2]
                  for r in selected])[..., None]
    y = np.asarray([r['label'] == 'drone' for r in selected], dtype=np.float32)
    net = build_cnn(42)
    optimizer = tf.keras.optimizers.Adam(.001, clipnorm=1.)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            logits = tf.reshape(net(x, training=True), [-1])
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
            objective = loss + tf.add_n(net.losses)
        gradients = tape.gradient(objective, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables, strict=True))
        return loss, tf.linalg.global_norm(gradients)

    forward = tf.function(lambda: tf.sigmoid(tf.reshape(net(x, training=False), [-1])))
    history = []
    last_gradient_norm = 0.
    started = time.perf_counter()
    for update in range(2001):
        if update % 50 == 0:
            p = forward().numpy()
            item = {'update': update, 'training_accuracy': float(np.mean((p >= .5) == y)),
                'training_bce': float(np.mean(-y * np.log(np.maximum(p, 1e-8)) - (1-y) * np.log(np.maximum(1-p, 1e-8)))),
                'min_correct_class_score': float(np.min(np.where(y, p, 1-p))),
                'seconds': time.perf_counter() - started}
            if update:
                item['last_gradient_norm'] = last_gradient_norm
            history.append(item)
            print('Memorization:', item, flush=True)
            if item['training_accuracy'] == 1 and item['min_correct_class_score'] > .95:
                break
        if update < 2000:
            loss, gradient_norm = step()
            last_gradient_norm = float(gradient_norm)
            if not np.isfinite(float(loss)) or not np.isfinite(last_gradient_norm):
                raise ValueError('Nonfinite training loss or gradient')
    result = {'scope': 'Fit the SAME 16 real training contexts; mechanical optimization check only, NO generalization test',
        'labels': 'Publisher recording labels inherited by middle context, NOT independently verified audible events',
        'selection': 'First 8 drone, 4 helicopter and 4 background Svanstrom training recordings, sorted by id; one middle cached context each',
        'settings': 'Same compact1781 CNN seed42, peak-relative linear256 input, Adam .001 clipnorm1, dropout .1, L2 .0001; supervised singleton BCE, full batch16, max2000 updates, stop100% correct and minimum correct-class score>.95',
        'records': [{'id': r['id'], 'sha256': r['sha256'], 'label': r['label'],
                     'start_sample': r['starts'][len(r['starts']) // 2]} for r in selected],
        'history': history, 'scores': forward().numpy().astype(float).tolist()}
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / 'memorization-weights.npz', **{f'w{i}': w for i, w in enumerate(net.get_weights())})
    write_json(REPORT / 'memorization-control.json', result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('control', choices=['inputs', 'architecture', 'memorization'])
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        {'inputs': audit_inputs, 'architecture': architecture_control,
         'memorization': memorization_control}[args.control]()


if __name__ == '__main__':
    main()
