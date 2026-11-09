#!/usr/bin/env python3
"""Assess a prespecified frozen panel on unfamiliar publisher recordings.

This intentionally supports a new manifest without weakening the existing
training-manifest checks. It never tunes or writes model packages.
"""

import gzip
import time
from pathlib import Path

from threadpoolctl import threadpool_limits

from drone_detector.audio import decode
from drone_detector.common import canonical, file_hash, hardware, read_json, read_jsonl, revision, write_json
from drone_detector.model import load_model
from drone_detector.stream import replay_audio
from drone_detector.weak import recording_results, weak_metrics

ROOT = Path('reports/external-v3')


def verify_panel(plan):
    models = {}
    for arm in plan['models']:
        bundle = Path(arm['bundle'])
        if file_hash(bundle / 'model.json') != arm['model_json_sha256']:
            raise ValueError('Frozen model metadata changed')
        model = load_model(bundle, require_frozen=True)
        if model.model_id != arm['model_id'] or model.metadata['decision'] != arm['decision']:
            raise ValueError('Frozen model or decision changed')
        models[arm['name']] = model
    return models


def main():
    plan_path = ROOT / 'panel-plan.json'
    manifest = ROOT / 'recordings.jsonl'
    records = read_jsonl(manifest)
    if not records or read_json(ROOT / 'data-audit.json')['cross_corpus_matches']:
        raise ValueError('No eligible data or unresolved exact overlap')
    if not read_json(ROOT / 'recovered-cnn-parity.json')['cases']:
        raise ValueError('Verify recovered models on prior fixtures first')
    plan = read_json(plan_path)
    models = verify_panel(plan)
    output = ROOT / 'assessment'
    ledger = ROOT / 'assessment-use.json'
    if output.exists() or ledger.exists():
        raise ValueError('This source cohort is already consumed; do not silently re-assess')
    execution = {'plan_sha256': file_hash(plan_path), 'manifest_sha256': file_hash(manifest),
                 'compatibility_amendment_sha256': file_hash(ROOT / 'compatibility-amendment.json'),
                 'source_audit_sha256': file_hash(ROOT / 'data-audit.json'), 'code': revision(),
                 'runner_sha256': file_hash(__file__), 'hardware': hardware(),
                 'models': {name: model.model_id for name, model in models.items()},
                 'scope': 'New-source recording-label assessment. Source/session independence and audible truth have the limitations stated in the audit. No field-reliability claim.'}
    output.mkdir()
    write_json(output / 'frozen-execution.json', execution)
    write_json(ledger, execution)  # Consume before any waveform is scored.
    report = {'execution': execution, 'models': {}}
    for name, model in models.items():
        scores, parity = {}, None
        started = time.perf_counter()
        with gzip.open(output / f'{name}.scores.jsonl.gz', 'wb') as stream:
            for index, row in enumerate(records):
                path = Path(row['path'])
                if file_hash(path) != row['sha256']:
                    raise ValueError('Assessment audio bytes changed')
                audio, _ = decode(path, model.config.sample_rate)
                if hasattr(model, 'batch_trace'):
                    trace = model.batch_trace(audio)
                    if index == 0:
                        # Independent sequential path on the same complete recording.
                        sequential = replay_audio(model, audio)
                        assert len(trace) == len(sequential)
                        errors = [abs(a['score'] - b['score']) for a, b in zip(trace, sequential, strict=True) if a['score'] is not None]
                        delta = max(errors, default=0)
                        assert delta <= 1e-5
                        assert [t['alarm'] for t in trace] == [t['alarm'] for t in sequential]
                        parity = {'recording_id': row['id'], 'windows': len(trace), 'max_score_error': delta, 'passed': True}
                else:
                    trace = replay_audio(model, audio)
                scores[row['id']] = trace
                stream.write(canonical({'recording_id': row['id'], 'model_id': model.model_id, 'trace': trace}) + b'\n')
                if (index + 1) % 10 == 0:
                    print(name, index + 1, '/', len(records), 'elapsed', round(time.perf_counter() - started, 1), flush=True)
        outcomes = recording_results(records, scores, model.metadata['decision'])
        sources = sorted({r['source'] for r in records})
        breakdown = {}
        for source in sources:
            selected = [r for r in outcomes if r['source'] == source]
            eligible = [r for r in selected if r['duration_seconds'] >= model.config.window_seconds + (model.metadata['decision']['on_hits'] - 1) * model.config.hop_seconds]
            breakdown[source] = {'all_replay_outcomes': weak_metrics(selected),
                'full_context_alarm_eligible': weak_metrics(eligible) if eligible else None,
                'excluded_short_recordings': len(selected) - len(eligible),
                'interpretation': 'Full-context alarm opportunity requires3s plus the existing confirmation hops. Single padded scores are retained as product behavior, not comparable FPR evidence.'}
        coverage = {'windows': sum(len(t) for t in scores.values()),
                    'scored_windows': sum(r['reason'] == 'scored' for t in scores.values() for r in t),
                    'short_recordings': [r['id'] for r in records if r['duration_seconds'] < model.config.window_seconds]}
        result = {'model_id': model.model_id, 'source_model_id': model.metadata.get('recovery', {}).get('source_model_id', model.model_id),
                  'decision': model.metadata['decision'], 'by_source': breakdown,
                  'seconds': time.perf_counter() - started, 'batch_sequential_parity': parity,
                  'coverage': coverage, 'recordings': [{k: v for k, v in row.items() if k != 'trace'} for row in outcomes]}
        write_json(output / f'{name}.json', result)
        report['models'][name] = {k: v for k, v in result.items() if k != 'recordings'}
        print('COMPLETE', name, breakdown, flush=True)
    write_json(output / 'comparison.json', report)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
