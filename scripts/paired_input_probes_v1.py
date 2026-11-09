#!/usr/bin/env python3
"""Disclosed structural probes on already consumed, fixed real contexts."""

from pathlib import Path

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from cross_condition_v1 import fixed_models
from drone_detector.audio import decode
from drone_detector.common import file_hash, read_json, read_jsonl, write_json
from drone_detector.spectrogram import SpectrogramConfig, spectrogram
from transfer_diagnosis_v1 import tensor_probes

ROOT = Path('reports/cross-condition-v1')


def main():
    plan = read_json(ROOT / 'plan.json')
    models = fixed_models(plan)
    old = {row['id']: row for row in read_jsonl(ROOT / 'transfer-recordings.jsonl')}
    cases = [old[name] for name in ['fpv-fpv5-fpv_53_1', 'fpv-no_drone-ambient_kiili_3', 'fpv-no_drone-grass_trimmer_1']]
    cases += read_jsonl(ROOT / 'kenya-recordings.jsonl')
    data = Path('runs/external-v3/downloads/itu/itu-aris-lab-acoustic-drone-dataset')
    items = read_json(data / 'groups.json')['segments']
    assignment = read_json(data / 'splits/split_interval_disjoint.json')['assignment']
    for label in ['drone', 'background']:
        item = next(row for row in items if row['label'] == label and assignment[row['file']] == 'test')
        cases.append({'id': item['file'], 'path': str(data / item['file']), 'label': label, 'sha256': file_hash(data / item['file'])})
    selection = {'timing': 'Disclosed follow-up after transfer outcomes, all sources already consumed. No Nanjing audio or model results are used.',
                 'cases': [{key: row[key] for key in ['id', 'path', 'label', 'sha256']} for row in cases],
                 'context_rule': 'First and middle full1s context, middle on0.5s grid; duplicate start removed for1s ITU clips.',
                 'transformations': 'Original, uniform waveformgain0.25, time-mean repeated, ten common-time permutations, ten independent-band-time permutations. Fixed seed1000+case/context index.',
                 'interpretation': 'Nonphysical tensor permutations test numerical dependence, not new acoustic accuracy or an assumption that temporal motion is required for hovering drones.'}
    write_json(ROOT / 'paired-probe-plan.json', selection)
    results = []
    for index, row in enumerate(cases):
        if file_hash(row['path']) != row['sha256']:
            raise ValueError('Fixed case changed')
        audio, _ = decode(row['path'])
        middle = max(0, (len(audio) - 16000) // 16000) * 8000
        for start in sorted({0, middle}):
            wave = audio[start:start + 16000]
            assert len(wave) == 16000
            images_by_rule = {}
            for arm in plan['models']:
                cfg = SpectrogramConfig(**arm['features'])
                rule = arm['name'].split('-seed')[0].removesuffix('-logistic')
                if rule not in images_by_rule:
                    image = spectrogram(wave, cfg)
                    variants = {'original': [image], 'uniform_gain': [spectrogram(wave * .25, cfg)],
                                **tensor_probes(image, seed=1000 + index * 2 + int(start > 0))}
                    images_by_rule[rule] = variants
                variants = images_by_rule[rule]
                batch = np.stack([image for values in variants.values() for image in values])
                model = models[arm['name']]
                if arm['family'] == 'cnn':
                    logits = model.network()(batch[..., None], training=False).numpy().ravel().astype(float)
                else:
                    features = np.concatenate([batch.mean(axis=2), batch.std(axis=2)], axis=1).astype(np.float64)
                    logits = ((features - model['mean']) / model['scale']) @ model['coef'] + model['intercept'][0]
                cursor, output = 0, {}
                for name, values in variants.items():
                    part = logits[cursor:cursor + len(values)]
                    output[name] = {'logits': part.tolist(), 'scores': expit(part).tolist()}
                    cursor += len(values)
                if arm['family'] == 'cnn':
                    assert abs(output['original']['scores'][0] - model.score(wave)) < 1e-5
                results.append({'id': row['id'], 'label': row['label'], 'start_sample': start,
                                'model': arm['name'], 'variants': output})
    summaries = {}
    for arm in plan['models']:
        rows = [row for row in results if row['model'] == arm['name']]
        summary = {}
        for name in rows[0]['variants']:
            changed = [np.mean(np.abs(np.array(row['variants'][name]['logits']) - row['variants']['original']['logits'][0])) for row in rows]
            flips = [np.mean((np.array(row['variants'][name]['scores']) >= .5) != (row['variants']['original']['scores'][0] >= .5)) for row in rows]
            summary[name] = {'mean_absolute_logit_change_equal_context_mass': float(np.mean(changed)),
                             'mean_classification_flip_fraction_equal_context_mass': float(np.mean(flips))}
        summaries[arm['name']] = summary
    write_json(ROOT / 'paired-input-probes.json', {'plan_sha256': file_hash(ROOT / 'paired-probe-plan.json'),
               'code_sha256': file_hash(__file__), 'results': results, 'summary': summaries,
               'scope': 'Already consumed selected contexts, not independent trials, global invariance or acoustic causal evidence. Raw logits avoid probability saturation hiding changes.'})
    print('Probed', len(results), 'model/context pairs; all transformations and raw logits retained')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
