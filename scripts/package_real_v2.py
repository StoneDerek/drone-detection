#!/usr/bin/env python3
"""Retain results, packages and disclosed demo inputs after the frozen assessment."""

import shutil
import tarfile
from pathlib import Path

from drone_detector.common import file_hash, read_json, read_jsonl, write_json


def main():
    root, reports, artifacts = Path('runs/real-v2'), Path('reports/real-v2'), Path('artifacts/real-v2')
    result = read_json(root / 'results.json')
    selection = read_json(root / 'frozen-selection.json')
    rows = read_jsonl(root / 'corpus/manifest-v2.jsonl')
    by_id = {r['id']: r for r in rows}
    selected = selection['selected']
    selected_id = selection['models'][selected]
    candidate = next(r for r in result['results'] if r['model_id'] == selected_id)
    reports.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    for name in ['results.json', 'experiment-plan.json', 'experiment-ledger.json',
                 'frozen-selection.json', 'data-plan.json', 'ddl-fetch.json']:
        shutil.copyfile(root / name, reports / name)
    shutil.copyfile(root / 'corpus/manifest-v2.jsonl', reports / 'manifest-snapshot.jsonl')
    shutil.copyfile(root / 'corpus/manifest-v2.weak-final-use.json', reports / 'final-use.json')
    for source, label in [(root / 'final', 'final'), (Path('runs/real-v1/final'), 'failed-v1-assessment')]:
        with tarfile.open(reports / (label + '-score-traces.tar.gz'), 'w:gz') as archive:
            for file in sorted(source.rglob('*')):
                if not file.is_file():
                    continue
                if file.name.endswith('.scores.jsonl'):
                    archive.add(file, arcname=str(file.relative_to(source)))
                else:
                    destination = reports / label / file.relative_to(source)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(file, destination)
    for name in ['ddl-quality-exclusions.json', 'evaluated-source.json']:
        shutil.copyfile(root / name, reports / name)
    for name, model in [('baseline', 'relative'), ('candidate', selected)]:
        destination = artifacts / name
        if destination.exists():
            raise ValueError('Do not overwrite packaged models')
        shutil.copytree(root / 'models' / model, destination)
    fixed = read_json(root / 'fixed-demo.json')
    failures = [r['id'] for r in candidate['records'] if r['truth'] != r['predicted']
                and by_id[r['id']]['duration_seconds'] <= 600]
    appendix = [identity for identity in failures if identity not in fixed['recording_ids']][:4]
    demo = {**fixed, 'recording_ids': fixed['recording_ids'] + appendix,
            'posthoc_failure_ids': appendix,
            'reason': fixed['reason'] + ' A separately labeled appendix adds the first four observed candidate errors after assessment.',
            'titles': {r['id']: r['source'] + ' · ' + (r.get('negative_kind') or r.get('drone_type') or r['label'])
                       for r in rows if r['id'] in fixed['recording_ids'] + appendix}}
    write_json(reports / 'demo-selection.json', demo)
    lines = ['# Real-audio product assessment v2', '',
        'Recording-level results against publisher labels. No independently verified event timing or field-reliability claim.', '',
        'The first real assessment failed and was explicitly consumed for development. It was not reused as a fresh test. See docs/real-data-investigation.md.', '',
        '## Data', '',
        '| Split | Files | Positive files | Negative files | Provenance/day groups | Audio hours |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for split in ['train', 'development', 'final']:
        subset = [r for r in rows if r['split'] == split]
        lines.append(f"| {split} | {len(subset)} | {sum(r['label']=='drone' for r in subset)} | {sum(r['label']=='non_drone' for r in subset)} | {len({r['group_id'] for r in subset})} | {sum(r['duration_seconds'] for r in subset)/3600:.3f} |")
    lines += ['', 'File counts are not independent trials. DDL excerpts share their original flights and capture day; the Batear day is one group. The small number of positive capture days limits uncertainty and operating-range claims.', '',
        '## Frozen comparison', '',
        f"Selected on development: **{selected}**, model `{selected_id}`.", '',
        '| Arm | Positive files detected | False-positive negative files | Recall | Negative-file FPR | Precision | Real-time factor |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for entry in result['results']:
        m = entry['summary']
        name = next(k for k, value in selection['models'].items() if value == entry['model_id'])
        lines.append(f"| {name} | {m['confusion']['tp']}/{m['positive_recordings']} | {m['confusion']['fp']}/{m['negative_recordings']} | {m['recording_recall']:.3f} | {m['negative_recording_false_positive_fraction']:.3f} | {m['precision'] if m['precision'] is not None else 'n/a'} | {entry['runtime']['real_time_factor']:.4f} |")
        lines += []
    lines += ['', '## Selected model by source', '',
        '| Source | Files | Positive files detected | False-positive negative files |',
        '| --- | ---: | ---: | ---: |']
    for source, m in candidate['source_breakdown'].items():
        lines.append(f"| {source} | {m['recordings']} | {m['confusion']['tp']}/{m['positive_recordings']} | {m['confusion']['fp']}/{m['negative_recordings']} |")
    m = candidate['summary']
    lines += ['', '## Limits and failures', '',
        f"Frozen decision: `{candidate['decision']}`. Group-bootstrap intervals: `{m['group_bootstrap_95']}`.", '',
        'The provisional recording targets are recall at least 0.9 and negative-file FPR at most 0.05. '
        f"Observed checks: recall **{'PASS' if m['recording_recall'] >= .9 else 'FAIL'}**, negative-file FPR **{'PASS' if m['negative_recording_false_positive_fraction'] <= .05 else 'FAIL'}**. "
        'These point checks do not fulfill the original event recall/FA-per-hour/delay contract, which remains unassessed.', '',
        'Observed errors (all are retained in JSON; first examples below):', '']
    for identity in failures[:15]:
        row = next(r for r in candidate['records'] if r['id'] == identity)
        lines.append(f"- `{identity}` / {row['source']}: publisher label {by_id[identity]['label']}; {'alarm' if row['predicted'] else 'no alarm'}; maximum score {row['max_score']:.4f}.")
    if not failures:
        lines.append('No errors observed on these publisher labels; this does not establish field reliability.')
    lines += ['', 'Class labels describe recordings, not independently listened-to drone events. Flight metadata and GPS distance do not establish acoustic audibility. No supported microphone, distance or SNR range is claimed. Event recall, false alarm events per true negative hour, physical capture delay and live-microphone operation require separately annotated capture.', '',
        'Next useful work: independent full recordings from additional days/devices, with audible onsets/offsets and long hard-negative sessions. Keep this assessment consumed if its errors guide further model optimization.', '',
        'The app supports real file upload, full-recording scores, playback/seeking, alarm intervals, input-quality flags and JSON export. It is recorded replay. The demo includes metadata-selected positives/ordinary/hard negatives and a labeled observed-error appendix.', '']
    (reports / 'RESULTS.md').write_text('\n'.join(lines))
    checksums = sorted(p for p in artifacts.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (artifacts / 'SHA256SUMS').write_text(''.join(f'{file_hash(p)}  {p.relative_to(artifacts)}\n' for p in checksums))
    print('Retained real model packages, source-level results, all traces and disclosed demo inputs.')


if __name__ == '__main__':
    main()
