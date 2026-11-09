#!/usr/bin/env python3
"""Preserve the coarse sweep; add exact recording-budget DEVELOPMENT calibration."""

import gzip
import json
import tarfile
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from drone_detector.common import canonical, digest, file_hash, hardware, read_json, revision, write_json
from drone_detector.data import checked_audio, load_manifest
from drone_detector.model import load_model, save_model
from drone_detector.spectral_calibration import choose_budget_trial, exact_budget_trials, freeze_calibration_metadata
from drone_detector.stream import replay_audio
from drone_detector.weak import recording_results, weak_metrics


def main():
    root = Path('runs/spectral-lab-v1')
    manifest = Path('runs/real-v2/corpus/manifest-v2.jsonl')
    plan = {'id': 'spectral-exact-development-budget-v1', 'code': revision(),
        'manifest_sha256': file_hash(manifest),
        'scope': 'Additional DEVELOPMENT calibration after observing coarse-grid failures. No new final assessment; no neural weights retrained; default unchanged.',
        'reason': 'Scores of small CNNs cluster between coarse grid thresholds. Separate a calibration limitation from spectral ranking.',
        'policy': 'For each on-hit count 1/2, set threshold to the (floor(.05*Nneg)+1)th largest negative recording activation statistic plus a 1e-5 numerical margin. Select maximal development recall within the 5% negative-file budget, then lower threshold, lower FPR, two hits. Off threshold .75×on, two off hits.',
        'warning': 'Development FPR is constrained by construction, not evidence of generalization. Budget is a provisional engineering choice.',
        'batching': 'Independent per-context features and batched classifier forward pass; sequential versus batch scores checked on first development recording. Audio timestamps are retained; offline batch traces claim no per-window delivery time.'}
    plan_path = root / 'calibration-plan.json'
    if not plan_path.exists():
        write_json(plan_path, plan)
        with tarfile.open(root / 'calibration-source.tar.gz', 'w:gz') as archive:
            for file in sorted(Path('src').rglob('*.py')):
                archive.add(file, arcname=str(file))
            archive.add('scripts/calibrate_spectral_models.py')
    plan = read_json(plan_path)
    implementation = revision()
    revisions = root / 'calibration-revisions'
    revisions.mkdir(exist_ok=True)
    snapshot = revisions / (implementation['source_sha256'] + '.tar.gz')
    if not snapshot.exists():
        with tarfile.open(snapshot, 'w:gz') as archive:
            for file in sorted(Path('src').rglob('*.py')):
                archive.add(file, arcname=str(file))
            archive.add('scripts/calibrate_spectral_models.py')
        write_json(snapshot.with_suffix('.json'), {'code': implementation, 'calibration_plan_sha256': digest(plan)})
    records = [r for r in load_manifest(manifest, True) if r['split'] == 'development']
    summary = read_json(root / 'calibrated-results.json') if (root / 'calibrated-results.json').exists() else {'plan': plan, 'models': {}}
    with threadpool_limits(limits=1):
        bundles = list((root / 'models').iterdir())
        if (root / 'followup-models').exists():
            bundles += list((root / 'followup-models').iterdir())
        for bundle in sorted(bundles, key=lambda value: value.name):
            if not (bundle / 'model.json').exists():
                continue
            name = bundle.name
            destination = root / 'calibrated' / name
            if destination.exists():
                if name not in summary['models']:
                    raise ValueError('Model exists without matching calibration report')
                continue
            model = load_model(bundle)
            scoring_decision = model.metadata.get('decision') or {'on_threshold': .5, 'off_threshold': .375, 'on_hits': 2, 'off_hits': 2}
            scores, parity = {}, None
            begin = time.perf_counter()
            trace_dir = root / 'calibrated-traces'
            trace_dir.mkdir(exist_ok=True)
            trace_file = trace_dir / (name + '.jsonl.gz')
            with gzip.open(trace_file, 'wb') as stream:
                for index, record in enumerate(records):
                    audio = checked_audio(record, manifest, model.config.sample_rate)
                    trace = model.batch_trace(audio, scoring_decision)
                    scores[record['id']] = trace
                    stream.write(canonical({'recording_id': record['id'], 'trace': trace}) + b'\n')
                    if index == 0:
                        sequential = replay_audio(model, audio, scoring_decision)
                        assert len(trace) == len(sequential)
                        errors = []
                        for a, b in zip(trace, sequential, strict=True):
                            for key in ['window_start_seconds', 'timestamp_seconds', 'reason', 'padded', 'quality_flags']:
                                assert a[key] == b[key], key
                            if a['score'] is not None:
                                errors.append(abs(a['score'] - b['score']))
                        maximum = max(errors, default=0.)
                        assert maximum <= 1e-5, maximum
                        parity = {'recording_id': record['id'], 'windows': len(trace), 'max_score_error': maximum,
                                  'tolerance': 1e-5, 'passed': True}
                    if (index + 1) % 40 == 0:
                        print(f'Exact development calibration {name}: {index + 1}/{len(records)}', flush=True)
            elapsed = time.perf_counter() - begin
            trials = exact_budget_trials(records, scores)
            chosen = choose_budget_trial(trials)
            contract = {'id': plan['id'], 'scope': plan['scope'], 'selection': plan['policy'],
                        'label_scope': 'Publisher recording labels; not audible-event truth', 'no_field_reliability_claim': True}
            selection = {'split': 'development', 'trials': trials, 'chosen': chosen,
                         'runtime_seconds': elapsed, 'code': implementation, 'calibration_plan_sha256': digest(plan)}
            metadata = freeze_calibration_metadata(model.metadata, chosen['decision'], contract, selection)
            calibrated = save_model(destination, metadata, model.arrays)
            for key in model.arrays:
                assert np.array_equal(calibrated.arrays[key], model.arrays[key])
            outcomes = recording_results(records, scores, chosen['decision'])
            source_breakdown = {source: weak_metrics([r for r in outcomes if r['source'] == source])
                                for source in sorted({r['source'] for r in outcomes})}
            report = {'model_id': calibrated.model_id, 'parent_model_id': model.model_id,
                'family': calibrated.metadata['family'], 'features': model.metadata['features'],
                'neural_parameters_unchanged': True, 'development': chosen,
                'source_breakdown': source_breakdown, 'group_uncertainty': weak_metrics(outcomes, intervals=True)['group_bootstrap_95'],
                'batch_parity': parity, 'runtime_seconds': elapsed, 'hardware': hardware(),
                'trace_sha256': file_hash(trace_file),
                'records': [{k: v for k, v in row.items() if k != 'trace'} for row in outcomes]}
            summary['models'][name] = report
            write_json(root / 'calibrated-results.json', summary)
            print(f'Calibrated {name}: {json.dumps(chosen)}', flush=True)


if __name__ == '__main__':
    main()
