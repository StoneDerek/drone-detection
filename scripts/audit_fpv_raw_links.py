#!/usr/bin/env python3
"""Check provided interval files against the separately supplied raw-view members."""

import argparse
import collections
import zipfile
from pathlib import Path, PurePosixPath

from drone_detector.common import file_hash, read_json, write_json, write_jsonl
from drone_detector.fpv_lineage import ExactNativeIndex, original_intervals
from audit_fpv_lineage import read_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('work/fpv-lineage-v1'))
    parser.add_argument('--metadata', type=Path, default=Path('work/positive-source-discovery-v1'))
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    previous = read_json(args.work / 'summary.json')
    if file_hash(args.archive) != previous['archive']['sha256']:
        raise ValueError('Verified archive changed')
    entries = read_json(args.metadata / 'Fpv.zip.directory.json')
    raw = [r for r in entries if '/labeled_data/' not in r['path']
           and PurePosixPath(r['path']).suffix.lower() in {'.wav', '.m4a'}]
    indexes = collections.defaultdict(list)
    raw_metadata, outcomes = [], []
    with zipfile.ZipFile(args.archive) as archive:
        for row in raw:
            audio, rate, meta = read_audio(archive, row['path'])
            p = PurePosixPath(row['path'])
            indexes[p.parts[1], p.parent.name.casefold()].append(
                (row['path'], rate, meta, ExactNativeIndex(audio))
            )
            raw_metadata.append(dict(member=row['path'], **meta))
        for row in original_intervals(entries):
            member = row['path']
            p = PurePosixPath(member)
            candidates = indexes.get((p.parts[2], p.parts[-3].casefold()), [])
            if not candidates:
                outcomes.append(dict(member=member, status='matching_raw_view_not_provided', matches=[]))
                continue
            audio, rate, info = read_audio(archive, member)
            matches, ambiguous = [], []
            for raw_member, raw_rate, meta, index in candidates:
                if raw_rate != rate:
                    continue
                result = index.match(audio)
                if result['status'] == 'unique_exact_match':
                    matches.append(dict(raw_member=raw_member, **result,
                                        byte_identical=meta['sha256'] == info['sha256']))
                elif result['status'] == 'ambiguous_repeated_waveform':
                    ambiguous.append(raw_member)
            status = ('unique_exact_raw_match' if len(matches) == 1 and not ambiguous else
                      'ambiguous_raw_match' if matches or ambiguous else 'no_exact_raw_match')
            outcomes.append(dict(member=member, status=status, matches=matches, ambiguous_raw_members=ambiguous))
    write_jsonl(args.work / 'raw-members.jsonl', raw_metadata)
    write_jsonl(args.work / 'interval-to-raw.jsonl', outcomes)
    result = dict(
        raw_audio_members_accounted=len(raw_metadata), original_intervals_accounted=len(outcomes),
        status_counts=dict(collections.Counter(r['status'] for r in outcomes)),
        byte_identical_interval_raw_pairs=sum(m['byte_identical'] for r in outcomes for m in r['matches']),
        scope='Exact decoded sample links to available raw members from the same declared capture/device. A missing match is not a global search across unrelated filenames or views. No raw labels are invented.',
        detector_scores_generated=0,
    )
    write_json(args.work / 'raw-links-summary.json', result)
    print(result)


if __name__ == '__main__':
    main()
