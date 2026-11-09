#!/usr/bin/env python3
"""Adaptive, disclosed input-band sensitivity follow-up on the fixed cases."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from drone_detector.audio import decode
from drone_detector.common import file_hash, read_json, read_jsonl, write_json
from drone_detector.model import load_model
from drone_detector.spectrogram import axes, spectrogram


def main():
    root = Path('reports/transfer-diagnosis-v1')
    plan = read_json(root / 'occlusion-followup-plan.json')
    assert plan['parent_cases_sha256'] == file_hash(root / 'cases.json')
    models = {a['name']: load_model(a['bundle'], True) for a in read_json('reports/external-v3/panel-plan.json')['models']
              if a['name'].startswith('coherent-')}
    records = {row['id']: row for row in read_jsonl('reports/external-v3/recordings.jsonl')}
    config = next(iter(models.values())).config
    frequencies, _, _ = axes(config)
    starts = list(range(0, 256 - 32 + 1, 16))
    results = []
    for case in read_json(root / 'cases.json'):
        wave, _ = decode(records[case['id']]['path'], 16000)
        start = round(case['start_seconds'] * 16000)
        wave = wave[start:start + 48000]
        image = spectrogram(wave, config)
        variants = []
        for low in starts:
            changed = image.copy()
            changed[low:low + 32] = 0
            variants.append(changed)
        row = {'id': case['id'], 'context': case['context'], 'start_seconds': case['start_seconds'], 'models': {}}
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        for name, model in models.items():
            actual = model.score(wave)
            assert abs(actual - case['scores'][name]) <= 1e-5
            scores = model.score_images(np.stack(variants))
            data = [{'lower_hz': float(frequencies[low]), 'upper_hz': float(frequencies[low + 31]), 'score': float(score)}
                    for low, score in zip(starts, scores, strict=True)]
            row['models'][name] = {'original_score': actual, 'occlusions': data}
            centers = [(.5 * (item['lower_hz'] + item['upper_hz'])) for item in data]
            line = ax.plot(centers, scores, marker='o', label=name)[0]
            ax.axhline(actual, color=line.get_color(), linestyle='--', alpha=.6)
        ax.set(xlabel='Center frequency of32-row occlusion (Hz)', ylabel='Classifier score', ylim=(-.03, 1.03),
               title=f'{case["id"]}: {case["context"]}\nSolid: numerical input-band removal; dashed: original score')
        ax.legend(fontsize=8)
        fig.savefig(root / f'{case["id"]}-{case["context"]}-occlusion.png', dpi=130)
        plt.close(fig)
        results.append(row)
        print(case['id'], case['context'], {name: (round(item['original_score'], 3),
              round(min(d['score'] for d in item['occlusions']), 3)) for name, item in row['models'].items()}, flush=True)
    write_json(root / 'occlusion.json', {'plan_sha256': file_hash(root / 'occlusion-followup-plan.json'),
               'cases': results, 'interpretation': plan['limits']})


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
