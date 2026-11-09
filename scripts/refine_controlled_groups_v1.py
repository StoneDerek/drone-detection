#!/usr/bin/env python3
"""Metadata-only grouping correction, declared before first model scores."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from controlled_source_v1 import csv_rows
from drone_detector.common import file_hash, read_json, read_jsonl, write_json, write_jsonl

ROOT = Path('reports/cross-condition-v1/controlled-source')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['freeze', 'apply'])
    args = parser.parse_args()
    if (ROOT / 'first-use.json').exists():
        raise ValueError('Grouping must be resolved before first model scores')
    sessions = {row['session_id']: row for row in csv_rows('sessions')}
    records = csv_rows('recordings')
    assignment = {row['recording_id']: ('nanjing-undocumented-historical-capture'
                   if not sessions[row['session_id']]['acquisition_date']
                   else 'nanjing-' + sessions[row['session_id']]['acquisition_date']) for row in records}
    path = ROOT / 'grouping-amendment.json'
    if args.phase == 'freeze':
        if path.exists():
            raise ValueError('Preserve the existing metadata-only amendment')
        write_json(path, {'created_utc': datetime.now(timezone.utc).isoformat(),
            'parent_plan_sha256': file_hash(ROOT / 'plan.json'),
            'session_table_sha256': file_hash(ROOT / 'source-records/sessions.csv'),
            'code_sha256': file_hash(__file__), 'assignment': assignment,
            'reason': 'Session table explicitly marks212 historical dates unavailable, including all200 positives. Date-like recording/batch IDs and JSON dates do not establish independent capture dates.',
            'change': 'Supersedes only the original plan grouping. Pool all212 undocumented historical captures, positive and negative, together. Use explicitly dated session table dates for the remaining61. Three conservative metadata components, one site/device family; no independence estimate.',
            'unchanged': 'All273 selected physical recordings, channel0, all12 fixed models/thresholds, sampling and recording-level label scope. No source model scores have been inspected.'})
        print('GROUPS FROZEN', dict(Counter(assignment.values())))
    else:
        amendment = read_json(path)
        if assignment != amendment['assignment'] or amendment['parent_plan_sha256'] != file_hash(ROOT / 'plan.json'):
            raise ValueError('Metadata grouping identity changed')
        rows = read_jsonl(ROOT / 'recordings.jsonl')
        for row in rows:
            row['publisher_batch_group'] = row['group_id']
            row['group_id'] = assignment[row['id']]
        write_jsonl(ROOT / 'recordings.jsonl', rows)
        audit = read_json(ROOT / 'data-audit.json')
        audit['grouping_amendment_sha256'] = file_hash(path)
        audit['conservative_acquisition_groups'] = len(set(assignment.values()))
        write_json(ROOT / 'data-audit.json', audit)
        print('GROUPS APPLIED', dict(Counter(row['group_id'] for row in rows)))


if __name__ == '__main__':
    main()
