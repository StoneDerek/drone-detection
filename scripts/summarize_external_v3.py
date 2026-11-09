#!/usr/bin/env python3
"""Reapply frozen decisions to retained traces and expose concrete failures."""

import gzip
import json
from collections import Counter
from pathlib import Path

from drone_detector.common import file_hash, read_json, read_jsonl, write_json
from drone_detector.weak import recording_results

ROOT = Path('reports/external-v3')


def main():
    comparison = read_json(ROOT / 'assessment/comparison.json')
    records = read_jsonl(ROOT / 'recordings.jsonl')
    diagnostics, checks, cases = {}, [], []
    for name in comparison['models']:
        original = read_json(ROOT / f'assessment/{name}.json')
        with gzip.open(ROOT / f'assessment/{name}.scores.jsonl.gz', 'rt') as stream:
            scores = {row['recording_id']: row['trace'] for row in map(json.loads, stream)}
        replay = recording_results(records, scores, original['decision'])
        expected = original['recordings']
        assert len(replay) == len(expected)
        for left, right in zip(replay, expected, strict=True):
            assert {k: v for k, v in left.items() if k != 'trace'} == right
        checks.append({'model': name, 'recordings': len(replay), 'outcomes_reproduced': True,
                       'trace_sha256': file_hash(ROOT / f'assessment/{name}.scores.jsonl.gz')})
        diagnostics[name] = {}
        for source in sorted({r['source'] for r in records}):
            eligible = [r for r in replay if r['source'] == source and r['duration_seconds'] >= 3.5]
            false_negatives = [r for r in eligible if r['truth'] and not r['predicted']]
            counts = Counter('all_scores_below_on_threshold' if r['max_score'] < original['decision']['on_threshold']
                             else 'peak_crossed_but_confirmation_not_met' for r in false_negatives)
            diagnostics[name][source] = {'false_negative_alarm_mechanisms': dict(counts)}
            for truth, predicted, kind in [(1, 0, 'false_negative'), (0, 1, 'false_positive'), (1, 1, 'detected_positive'), (0, 0, 'correct_negative')]:
                chosen = sorted([r for r in eligible if r['truth'] == truth and r['predicted'] == predicted], key=lambda r: r['id'])[:2 if truth != predicted else 1]
                for row in chosen:
                    trace = scores[row['id']]
                    peak = max(trace, key=lambda r: r['score'] if r['score'] is not None else -1)
                    record = next(r for r in records if r['id'] == row['id'])
                    cases.append({'model': name, 'source': source, 'kind': kind,
                        'recording_id': row['id'], 'publisher_label': record['label'],
                        'label_evidence': record['label_evidence'], 'audio_path': record['path'],
                        'audio_sha256': record['sha256'], 'duration_seconds': row['duration_seconds'],
                        'max_score': row['max_score'], 'decision': original['decision'],
                        'peak_context_start_seconds': peak['window_start_seconds'],
                        'peak_context_end_seconds': peak['timestamp_seconds'], 'alarms': row['alarms'],
                        'audibility_judgment': 'Not inferred from classifier score or publisher metadata.'})
    write_json(ROOT / 'trace-reproduction.json', {'checks': checks, 'passed': True,
        'scope': 'Reapplying frozen alarm rules to saved scores; not a new assessment.'})
    write_json(ROOT / 'failure-diagnostics.json', diagnostics)
    write_json(ROOT / 'review-cases.json', cases)
    lines = [
        '# Frozen external-source assessment v3', '',
        'Four pre-existing models were evaluated at their existing frozen decisions. No weights, amplitude rules or thresholds were tuned on these recordings. Results vary strongly by source; this assessment does not establish field reliability.', '',
        '| Frozen model | FPV positives detected /21 | FPV negative files alarmed /16 | ITU eligible positive intervals detected /34 |',
        '| --- | ---: | ---: | ---: |',
    ]
    for name, result in comparison['models'].items():
        fpv = result['by_source']['saarmets-fpv']['full_context_alarm_eligible']['confusion']
        itu = result['by_source']['itu-aris']['full_context_alarm_eligible']['confusion']
        lines.append(f'| {name} | {fpv["tp"]} | {fpv["fp"]} | {itu["tp"]} |')
    lines += [
        '', '## What those counts measure', '',
        'FPV contains 37 publisher-labelled recordings (21 positive, 16 negative), totaling 1.414 h. Its session ancestry is undocumented and conservatively forms one source component. These are not 37 independently verified capture sessions. Source attribution and limitations are in SOURCES.md.', '',
        'ITU contains 35 manually selected positive intervals from one original session, plus 1,000 one-second background clips. Only 34 positives provide a full 3 s context plus a second confirmation score. The remaining 1 s positive and all background files are retained in raw replay results but excluded from the comparable full-context cohort.', '',
        'Every model gives zero alarms on those one-second backgrounds because the frozen decision needs two hits. That is a buffering/confirmation consequence, not evidence of zero field false positives. There is no eligible continuous negative denominator for ITU. The raw max-score AUCs also mix long positive intervals with short padded negatives, so they are not treated as a fair matched-duration discriminator benchmark.', '',
        'The two sources were absent from the task-specific training, repeatedly reused development and consumed v2 assessment. No exact/raw, canonical-decoded or gain-normalized whole-file matches to the 2,356 earlier recordings were found. That cannot prove the absence of unknown shifted excerpts, lossy copies or all audio-pretraining overlap. Each source is one conservative provenance component, so file/window bootstrap intervals would overstate independent evidence.', '',
        'Labels are publisher evidence, not newly invented listening judgments. ITU reports two annotators and kappa 0.5869; its background comes from unlabelled stretches. Reviewed audible event boundaries, truly negative continuous stretches and independent new capture sessions remain missing. Event recall, true false alarms/hour, physical capture delay and range remain unassessed.', '',
        '## Concrete alarm failures', '',
    ]
    for name, sources in diagnostics.items():
        lines.append(f'- {name}: ' + '; '.join(f'{source}: {details["false_negative_alarm_mechanisms"]}' for source, details in sources.items()))
    lines += ['',
        'These mechanisms describe the frozen alarm engine: either all scores remain below its on-threshold, or isolated crossings fail confirmation. They do not diagnose whether an acoustic cue was audible or why a model assigned that score. review-cases.json retains the first two false positives/negatives by sorted original ID per source/model, plus one correct positive/negative where present. Each case includes its exact audio hash, peak context, decision and alarm intervals. Those examples were not selected to make a model look favorable.', '',
        '## Separate small published-label task', '',
        'The predeclared one-second non-neural task is documented in ../human-label-task-v1/RESULTS.md. It matches class durations, uses only complete STFT frames, train-only statistics and publisher interval-disjoint assignments. At fixed 0.5 it detects 10/10 test positive intervals and 0/3 background blocks, but only 183/674 positive-labelled seconds (AUC 0.888). This shows the importance of the target and aggregation: a single high-scoring second can satisfy an interval label. It is same-session feasibility, not a replacement for fresh-session validation or a direct model-versus-model comparison.', '',
        '## Reproducibility and next evidence', '',
        'panel-plan.json fixes model IDs, preprocessing and decisions before scoring. compatibility-amendment.json records the input-duration issue and the matched-duration task before external scoring. assessment/frozen-execution.json fixes the manifest, source audit, runner, hardware and plan hashes. assessment-use.json consumes this cohort before inference. The recovered CNN packages and 32 prior-fixture comparisons are described in RECOVERY.md.', '',
        'All compressed timestamped scores, per-recording outcomes and models are retained. Run scripts/summarize_external_v3.py to reproduce every saved recording outcome from the frozen decisions without scoring audio again. The one-use evaluation runner refuses a second assessment in place. Its numerical results are now consumed for any subsequent model optimization.', '',
        'The next discriminating evidence is another independently captured session with matched drone/background conditions, manually reviewed audible intervals and explicit ambiguous spans. The source-dependent reversals do not support declaring one architecture family inherently best. The fixed one-second task shows that this spectral input can carry useful published-label distinctions, while leaving cross-session generalization unresolved.', '',
    ]
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines))
    print('Reproduced', sum(check['recordings'] for check in checks), 'model/recording outcomes; selected', len(cases), 'prespecified review cases')
    for name, sources in diagnostics.items():
        print(name, sources)


if __name__ == '__main__':
    main()
