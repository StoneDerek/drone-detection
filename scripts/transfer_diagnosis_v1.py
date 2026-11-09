#!/usr/bin/env python3
"""Consumed-source score diagnosis and real-input structural probes."""

import gzip
import json
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from drone_detector.audio import decode
from drone_detector.common import file_hash, read_json, read_jsonl, write_json
from drone_detector.model import load_model
from drone_detector.spectral_calibration import activation_statistic
from drone_detector.spectrogram import axes, spectrogram
from drone_detector.weak import recording_results, weak_metrics

REPORT = Path('reports/transfer-diagnosis-v1')
EXTERNAL = Path('reports/external-v3')


def tensor_probes(image, seed=42, repeats=10):
    """Preserve each band's marginal values while changing time relationships."""
    rng = np.random.default_rng(seed)
    result = {'time_mean': [np.repeat(image.mean(axis=1, keepdims=True), image.shape[1], axis=1)]}
    common, independent = [], []
    for _ in range(repeats):
        common.append(image[:, rng.permutation(image.shape[1])])
        independent.append(np.stack([row[rng.permutation(len(row))] for row in image]))
    result.update(common_time_permutation=common, independent_band_time_permutation=independent)
    return result


def read_traces(path):
    with gzip.open(path, 'rt') as stream:
        return {row['recording_id']: row['trace'] for row in map(json.loads, stream)}


def decision_diagnosis(records, outcomes, traces):
    report = {}
    for name, result in outcomes.items():
        fpv = [r for r in records if r['source'] == 'saarmets-fpv']
        stats = [activation_statistic(traces[name][r['id']], 2) for r in fpv]
        y = np.array([r['label'] == 'drone' for r in fpv])
        cutoff = float(max(value for value, label in zip(stats, y, strict=True) if not label) + 1e-5)
        report[name] = {'two_hit_activation_auc': float(roc_auc_score(y, stats)),
            'within_fpv_zero_false_file_ceiling': {'threshold': cutoff if cutoff <= 1 else None,
                'feasible_probability_threshold': cutoff <= 1,
                'positive_files_detected': int(sum(value >= cutoff and label for value, label in zip(stats, y, strict=True))) if cutoff <= 1 else None,
                'warning': 'Oracle on consumed data only; finite16-negative budget allows0 false files. Not a selected deployment or fresh validation.'}}
    with tarfile.open('reports/cnn-foundations-v1/development-score-traces.tar.gz') as archive:
        for name in ['coherent-frequency-seed42', 'coherent-frequency-seed7']:
            with gzip.open(archive.extractfile(name + '.jsonl.gz'), 'rt') as stream:
                old_scores = {row['recording_id']: row['trace'] for row in map(json.loads, stream)}
            old = read_json(f'reports/cnn-foundations-v1/{name}-evaluation.json')['recordings']
            negative = sorted((activation_statistic(old_scores[row['id']], 2) for row in old if not row['truth']), reverse=True)
            old_minimum = negative[int(np.floor(.05 * len(negative)))] + 1e-5
            new_minimum = report[name]['within_fpv_zero_false_file_ceiling']['threshold']
            threshold = float(max(old_minimum, new_minimum))
            decision = {**outcomes[name]['decision'], 'on_threshold': threshold, 'off_threshold': .75 * threshold}
            old_results = recording_results(old, old_scores, decision)
            new_records = [r for r in records if r['source'] == 'saarmets-fpv']
            new_results = recording_results(new_records, traces[name], decision)
            report[name]['shared_threshold_control'] = {'decision': decision,
                'old_development': weak_metrics(old_results), 'consumed_fpv': weak_metrics(new_results),
                'change': 'Weights and all preprocessing fixed. One threshold respecting<=5% separately on188 earlier and16 FPV negatives. Off/on ratio and confirmation fixed.',
                'scope': 'Development-only bounded decision control; not an independent improvement.'}
    write_json(REPORT / 'decision-diagnosis.json', report)
    return report


def case_figure(case, wave, image, models, scores, interventions, destination):
    fig, panel = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    time = np.arange(len(wave)) / 16000
    panel[0, 0].plot(time, wave, lw=.35)
    panel[0, 0].set(xlabel='Time (s)', ylabel='PCM amplitude', title='Actual three-second waveform')
    f, t, z = signal.stft(wave, fs=16000, nperseg=4096, noverlap=3840, boundary=None, padded=False)
    db = 10 * np.log10(np.maximum(np.abs(z).astype(np.float64) ** 2, 1e-12))
    selected = (f >= 50) & (f <= 5000)
    plot = panel[0, 1].pcolormesh(t, f[selected], db[selected], vmin=-100, vmax=-10, shading='auto', cmap='magma')
    panel[0, 1].set(xlabel='Time (s)', ylabel='Hz', title='4096-point STFT; common numerical power reference')
    fig.colorbar(plot, ax=panel[0, 1], label='dB relative to unit STFT power')
    freq, times, _ = axes(models['coherent-frequency-seed42'].config)
    panel[0, 2].pcolormesh(times, freq, image, vmin=0, vmax=1, shading='auto', cmap='magma')
    panel[0, 2].set(xlabel='Time (s)', ylabel='Hz', ylim=(50, 5000), title='Actual CNN input; context-peak reference')
    for column, name in enumerate(['coherent-frequency-seed42', 'coherent-frequency-seed7']):
        cam = models[name].gradcam(wave, target='drone', layer='features')
        ax = panel[1, column]
        ax.pcolormesh(times, freq, image, vmin=0, vmax=1, shading='auto', cmap='gray')
        ax.imshow(cam['heatmap'], origin='lower', aspect='auto', extent=[times[0], times[-1], freq[0], freq[-1]], alpha=.55,
                  vmin=0, vmax=1, cmap='inferno', interpolation='nearest')
        ax.set(xlabel='Time (s)', ylabel='Hz', ylim=(50, 5000),
               title=f'{name}; score {scores[name]:.3f}\nGrad-CAM native {cam["coarse_shape"]}; positive map={cam["has_positive_attribution"]}')
    ax = panel[1, 2]
    names = ['original', 'uniform_gain', 'half_gain', 'time_mean', 'common_time_permutation', 'independent_band_time_permutation']
    labels = ['Original', 'Uniform ×.25', 'Second half ×.25', 'Time mean', 'Common shuffle', 'Per-band shuffle']
    for offset, name in [(-.12, 'coherent-frequency-seed42'), (.12, 'coherent-frequency-seed7')]:
        values = [[scores[name]]] + [interventions[name][key] for key in names[1:]]
        centers = np.array([np.mean(v) for v in values])
        low, high = np.array([min(v) for v in values]), np.array([max(v) for v in values])
        ax.errorbar(np.arange(len(names)) + offset, centers, yerr=[centers-low, high-centers], marker='o', capsize=3, label=name)
    ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel='Classifier score', ylim=(-.03, 1.03), title='Diagnostic interventions; error bars=range of10 shuffles')
    ax.tick_params(axis='x', labelrotation=30)
    ax.legend(fontsize=8)
    fig.suptitle(f'{case["id"]}: {case["context"]} context at {case["start_seconds"]:.1f}s — publisher label {case["label"]}', fontsize=14)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main():
    REPORT.mkdir(exist_ok=True)
    records = read_jsonl(EXTERNAL / 'recordings.jsonl')
    by_id = {r['id']: r for r in records}
    plan = read_json(EXTERNAL / 'panel-plan.json')
    models = {arm['name']: load_model(arm['bundle'], True) for arm in plan['models']}
    outcomes = {name: read_json(EXTERNAL / f'assessment/{name}.json') for name in models}
    traces = {name: read_traces(EXTERNAL / f'assessment/{name}.scores.jsonl.gz') for name in models}
    decision_diagnosis(records, outcomes, traces)
    predictions = {name: {r['id']: r['predicted'] for r in result['recordings']} for name, result in outcomes.items()}
    itu = next(r['id'] for r in sorted(records, key=lambda r: r['id']) if r['source'] == 'itu-aris' and r['duration_seconds'] >= 3.5
               and predictions['coherent-frequency-seed7'][r['id']] and not predictions['coherent-frequency-seed42'][r['id']])
    selected = ['fpv-fpv5-fpv_53_1', 'fpv-no_drone-ambient_kiili_3', 'fpv-no_drone-grass_trimmer_1', itu]
    result = []
    for number, identity in enumerate(selected):
        row = by_id[identity]
        assert file_hash(row['path']) == row['sha256']
        audio, _ = decode(row['path'], 16000)
        middle = int(np.floor(max(0, (len(audio) / 16000 - 3) / 2) / .5) * .5 * 16000)
        for context, start in [('first', 0), ('middle', middle)]:
            wave = audio[start:start + 48000]
            assert len(wave) == 48000
            half_gain = np.ones(len(wave), dtype=np.float32)
            center, taper = len(wave) // 2, 160
            half_gain[center:center + taper] = np.linspace(1, .25, taper)
            half_gain[center + taper:] = .25
            scores, interventions, invariance = {}, {}, {}
            for name, model in models.items():
                scores[name] = model.score(wave)
                values = {'uniform_gain': [model.score(wave * .25)], 'half_gain': [model.score(wave * half_gain)]}
                if hasattr(model, 'score_images'):
                    image = spectrogram(wave, model.config)
                    variants = tensor_probes(image, seed=1000 + number)
                    for label, images in variants.items():
                        values[label] = model.score_images(np.stack(images)).astype(float).tolist()
                    if name.startswith('svm'):
                        error = max(abs(v - scores[name]) for label in ['common_time_permutation', 'independent_band_time_permutation'] for v in values[label])
                        assert error < 1e-5
                        invariance[name] = error
                interventions[name] = values
            case = {'id': identity, 'source': row['source'], 'label': row['label'], 'label_evidence': row['label_evidence'],
                    'context': context, 'start_seconds': start / 16000, 'audio_sha256': row['sha256'],
                    'scores': scores, 'interventions': interventions, 'svm_permutation_max_score_error': invariance,
                    'meaning': 'Existing real input plus nonphysical tensor or gain probes. No invented audibility labels, causal recognition proof or accuracy trials.'}
            result.append(case)
            if context == 'middle':
                cnn_input = spectrogram(wave, models['coherent-frequency-seed42'].config)
                case_figure(case, wave, cnn_input, models, scores, interventions, REPORT / f'{identity}.png')
            print('CASE', identity, context, {name: round(value, 4) for name, value in scores.items()}, flush=True)
    write_json(REPORT / 'cases.json', result)
    print('COMPLETE structural case probes', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
