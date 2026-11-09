#!/usr/bin/env python3
"""Reapply retained decisions and check fixed real-input package scores."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.special import expit
from threadpoolctl import threadpool_limits

from cross_condition_v1 import fixed_models, source_metrics
from drone_detector.audio import decode
from drone_detector.common import file_hash, read_json, read_jsonl, write_json
from drone_detector.spectrogram import SpectrogramConfig, spectrogram

ROOT = Path('reports/cross-condition-v1')


def main():
    plan = read_json(ROOT / 'plan.json')
    models = fixed_models(plan)
    arms = {arm['name']: arm for arm in plan['models']}
    phases = [(ROOT / 'transfer', ROOT / 'transfer-recordings.jsonl'),
              (ROOT / 'kenya', ROOT / 'kenya-recordings.jsonl')]
    if (ROOT / 'controlled-source/comparison.json').exists():
        phases.append((ROOT / 'controlled-source', ROOT / 'controlled-source/recordings.jsonl'))
    checks, outcomes, legacy_outcomes = [], 0, 0
    for destination, manifest in phases:
        result = read_json(destination / 'comparison.json')
        records = {row['id']: row for row in read_jsonl(manifest)}
        # Prespecified software checks: first record in each source/class cell,
        # first and middle sampled context. These are repetitions, not trials.
        selected, cells = set(), set()
        for row in records.values():
            cell = (row['source'], row['label'])
            if cell not in cells:
                selected.add(row['id'])
                cells.add(cell)
        reproduced = {name: [] for name in result['models']}
        for trace_path in sorted(destination.glob('*.scores.jsonl.gz')):
            seen = set()
            with gzip.open(trace_path, 'rt') as stream:
                for line in stream:
                    trace = json.loads(line)
                    identifier = trace['recording_id']
                    assert identifier in records and identifier not in seen
                    seen.add(identifier)
                    starts = trace['start_samples']
                    assert len(starts) == len(set(starts)) and all(start >= 0 and start % 8000 == 0 for start in starts)
                    if destination.name != 'kenya':
                        assert len(starts) <= 60
                    assert len(trace['rms']) == len(starts)
                    for name, values in trace['scores'].items():
                        values = np.asarray(values)
                        assert len(values) == len(starts) and np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))
                        old = next(row for row in result['models'][name]['recordings'] if row['id'] == identifier)
                        new = {**old, 'predicted': int(bool(len(values) and np.any(values >= .5))),
                               'max_score': float(values.max()) if len(values) else None,
                               'positive_context_fraction': float(np.mean(values >= .5)) if len(values) else None,
                               'sampled_contexts': len(starts),
                               'silent_contexts': sum(rms < arms[name]['features']['silence_rms'] for rms in trace['rms'])}
                        assert new == old
                        reproduced[name].append(new)
                        outcomes += 1
                    if identifier in selected:
                        record = records[identifier]
                        assert file_hash(record['path']) == record['sha256']
                        wave, _ = decode(record['path'])
                        for i in sorted({0, len(starts) // 2}):
                            for name, values in trace['scores'].items():
                                arm, model = arms[name], models[name]
                                context = wave[starts[i]:starts[i] + 16000]
                                assert len(context) == 16000
                                if arm['family'] == 'cnn':
                                    score = model.score(context)
                                else:
                                    image = spectrogram(context, SpectrogramConfig(**arm['features']))
                                    features = np.concatenate([image.mean(axis=1), image.std(axis=1)]).astype(np.float64)
                                    score = float(expit(((features - model['mean']) / model['scale']) @ model['coef'] + model['intercept'][0]))
                                error = abs(float(values[i]) - score)
                                assert error < (1e-5 if arm['family'] == 'cnn' else 1e-8)
                                checks.append({'phase': destination.name, 'id': identifier, 'start_sample': starts[i], 'model': name, 'error': error})
            assert seen == set(records)
        for name, rows in reproduced.items():
            assert len(rows) == len(records)
            assert source_metrics(rows) == result['models'][name]['by_source']
        for name, legacy in result.get('legacy_original_policy', {}).items():
            with gzip.open(destination / f'legacy-{name}.jsonl.gz', 'rt') as stream:
                traces = {row['recording_id']: row['trace'] for row in map(json.loads, stream)}
            for row in legacy['recordings']:
                assert row['predicted'] == int(any(point['alarm'] for point in traces[row['id']]))
                assert row['max_score'] == max(point['score'] or 0 for point in traces[row['id']])
                legacy_outcomes += 1
            assert legacy['positive_files_detected'] == sum(row['predicted'] for row in legacy['recordings'])
    pcm_checks = 0
    if len(phases) == 3:
        source_plan = read_json(ROOT / 'controlled-source/plan.json')
        assert source_plan['models'] == plan['models']
        selected = {row['publisher']['recording_id']: row for row in source_plan['cohort']}
        amendment_path = ROOT / 'controlled-source/grouping-amendment.json'
        amendment = read_json(amendment_path)
        assert read_json(ROOT / 'controlled-source/data-audit.json')['grouping_amendment_sha256'] == file_hash(amendment_path)
        for row in read_jsonl(ROOT / 'controlled-source/recordings.jsonl'):
            assert row['original_wav_sha256'] == selected[row['id']]['wav']['sha256']
            assert row['original_json_sha256'] == selected[row['id']]['json']['sha256']
            assert row['group_id'] == amendment['assignment'][row['id']]
            assert file_hash(row['path']) == row['sha256']
            samples, rate = sf.read(row['path'], dtype='int16', always_2d=True)
            assert rate == 48000 and samples.shape[1] == 1
            assert hashlib.sha256(samples.astype('<i2').tobytes()).hexdigest() == row['int16_pcm_sha256']
            metadata = ROOT / 'controlled-source/recording-metadata' / (row['id'] + '.json')
            assert file_hash(metadata) == row['original_json_sha256']
            pcm_checks += 1
    report = {'code_sha256': file_hash(__file__), 'passed': True,
              'reproduced_classifier_recording_outcomes': outcomes, 'reproduced_legacy_alarm_outcomes': legacy_outcomes,
              'portable_real_input_checks': checks, 'maximum_portable_error': max(row['error'] for row in checks),
              'retained_48000Hz_lossless_pcm_and_publisher_json_checks': pcm_checks,
              'scope': 'Software reproducibility on already consumed inputs; these checks add no independent acoustic evidence.'}
    write_json(ROOT / 'verification.json', report)
    print('Verified', outcomes, 'classifier decisions;', legacy_outcomes, 'legacy alarms;', len(checks), 'direct real-input scores;', pcm_checks, 'PCM identities')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
