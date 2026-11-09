#!/usr/bin/env python3
"""Exact additive descriptor contrasts, with equal publisher-group/file mass.

Consumed-data explanation of the non-neural control, not an acoustic causal
intervention or a CNN attribution. All four amplitude policies are retained.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from cross_condition_v1 import context_starts
from drone_detector.audio import decode
from drone_detector.common import file_hash, read_json, read_jsonl, write_json
from drone_detector.spectrogram import SpectrogramConfig, spectrogram

REPORT = Path('reports/cross-condition-v1')
BANDS = np.array([50, 250, 500, 1000, 2000, 3000, 4000, 6000, 7600])


def descriptors(images):
    return np.concatenate([images.mean(axis=2), images.std(axis=2)], axis=1).astype(np.float64)


def contrast(rows, parameters):
    positive = np.mean([row['features'] for row in rows if row['positive']], axis=0)
    negative = np.mean([row['features'] for row in rows if not row['positive']], axis=0)
    terms = (positive - negative) / parameters['scale'] * parameters['coef']
    centers = np.linspace(50, 7600, 258)[1:-1]
    band_terms = [float(terms[:256][(centers >= lo) & (centers < hi)].sum() + terms[256:][(centers >= lo) & (centers < hi)].sum())
                  for lo, hi in zip(BANDS[:-1], BANDS[1:], strict=True)]
    return {'positive_groups': sum(row['positive'] for row in rows), 'negative_groups': sum(not row['positive'] for row in rows),
            'positive_mean_features': positive.tolist(), 'negative_mean_features': negative.tolist(),
            'feature_logit_contrast': terms.tolist(), 'band_logit_contrast': band_terms,
            'total_mean_logit_contrast': float(terms.sum())}


def main():
    data = Path('runs/external-v3/downloads/itu/itu-aris-lab-acoustic-drone-dataset')
    items = read_json(data / 'groups.json')['segments']
    assignment = read_json(data / 'splits/split_interval_disjoint.json')['assignment']
    fpv = [row for row in read_jsonl(REPORT / 'transfer-recordings.jsonl') if row['source'] == 'saarmets-fpv']
    plan = read_json(REPORT / 'plan.json')
    results = {}
    for arm in [row for row in plan['models'] if row['family'] == 'logistic']:
        name = arm['name'].removesuffix('-logistic')
        path = Path(arm['bundle']) / 'parameters.npz'
        if file_hash(path) != arm['parameters_sha256']:
            raise ValueError('Classifier changed')
        with np.load(path, allow_pickle=False) as archive:
            parameters = {key: archive[key].copy() for key in archive.files}
        images = np.load(f'runs/learnability-normalization-v1/{name}.npy', mmap_mode='r')
        grouped = {}
        for i, item in enumerate(items):
            if assignment[item['file']] == 'test':
                grouped.setdefault(item['group'], []).append(i)
        itu_rows = [{'positive': items[indices[0]]['label'] == 'drone', 'features': descriptors(images[indices]).mean(axis=0)}
                    for indices in grouped.values()]
        cfg = SpectrogramConfig(**arm['features'])
        fpv_rows = []
        for row in fpv:
            wave, _ = decode(row['path'])
            starts = context_starts(len(wave), 60)
            image = np.stack([spectrogram(wave[start:start + 16000], cfg) for start in starts])
            fpv_rows.append({'positive': row['label'] == 'drone', 'features': descriptors(image).mean(axis=0)})
        results[name] = {'itu_test_publisher_groups': contrast(itu_rows, parameters), 'fpv_recording_files': contrast(fpv_rows, parameters)}
        print(name, {key: round(value['total_mean_logit_contrast'], 3) for key, value in results[name].items()}, flush=True)
    write_json(REPORT / 'descriptor-contrast.json', {'code_sha256': file_hash(__file__),
        'scope': 'Consumed-data exact linear decomposition of mean logit positive-minus-negative. Each ITU publisher interval/background block or FPV file has equal mass within class; each of its contexts has equal mass. This explains a non-neural control, not CNN cues or audible truth. It is not the max-over-context decision statistic.',
        'band_edges_hz': BANDS.tolist(), 'results': results})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    labels = [f'{lo:g}–{hi:g}' for lo, hi in zip(BANDS[:-1], BANDS[1:], strict=True)]
    for ax, (name, result) in zip(axes.flat, results.items(), strict=True):
        x = np.arange(len(labels))
        for offset, (cohort, value) in zip([-.18, .18], result.items(), strict=True):
            ax.bar(x + offset, value['band_logit_contrast'], width=.36, label=cohort.split('_')[0].upper())
        ax.axhline(0, color='black', linewidth=.7)
        ax.set_title(name)
        ax.set_ylabel('Positive − negative mean logit contribution')
        ax.set_xticks(x, labels, rotation=35, ha='right')
        ax.legend()
    fig.suptitle('Same fixed non-neural classifier: frequency contrasts change across capture conditions')
    fig.text(.5, .01, 'Exact additive diagnostic; equal group/file mass. Band centers grouped in Hz. No new acoustic trials or CNN attribution.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(REPORT / 'descriptor-contrast.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
