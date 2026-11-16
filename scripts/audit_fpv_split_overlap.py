#!/usr/bin/env python3
"""Measure sample overlap only where native original/export matches are verified."""

import argparse
import collections
from pathlib import Path

from drone_detector.common import read_jsonl, write_json


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = result[-1][0], max(end, result[-1][1])
        else:
            result.append((start, end))
    return result


def overlap(first, second):
    first, second = merge(first), merge(second)
    left = right = total = 0
    while left < len(first) and right < len(second):
        total += max(0, min(first[left][1], second[right][1]) - max(first[left][0], second[right][0]))
        if first[left][1] < second[right][1]:
            left += 1
        else:
            right += 1
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('work/fpv-lineage-v1'))
    args = parser.parse_args()
    metadata = {r['member']: r for r in read_jsonl(args.work / 'decoded-members.jsonl')}
    lineage = {r['member']: r for r in read_jsonl(args.work / 'lineage.jsonl')}
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    matched = 0
    for row in read_jsonl(args.work / 'direct-export-to-original.jsonl'):
        if row['status'] != 'unique_exact_match':
            continue
        matched += 1
        groups[row['parent']][lineage[row['member']]['partition']].append(
            (row['start_frame'], row['start_frame'] + row['frames'])
        )
    pairs = []
    partitions = ['train', 'validate', 'test']
    for parent, parts in groups.items():
        for index, first in enumerate(partitions):
            for second in partitions[:index]:
                frames = overlap(parts[first], parts[second])
                if frames:
                    pairs.append(dict(parent=parent, partitions=[first, second], overlap_frames=frames,
                                      overlap_seconds=frames / metadata[parent]['sample_rate']))
    result = dict(
        verified_exports_accounted=matched, source_members_with_verified_exports=len(groups),
        source_members_with_multiple_verified_partitions=sum(sum(bool(v) for v in parts.values()) > 1 for parts in groups.values()),
        source_members_with_measured_cross_partition_overlap=len({r['parent'] for r in pairs}),
        overlap_pair_count=len(pairs),
        overlap_seconds_summed_by_source_member=sum(r['overlap_seconds'] for r in pairs),
        scope='Native-sample overlap only for uniquely matched original/export relationships. Unavailable/unmatched sources are unassessed. Sums across phone views are not independent capture hours.',
    )
    write_json(args.work / 'verified-split-overlap-summary.json', result)
    write_json(args.work / 'verified-split-overlaps.json', pairs)
    print(result)


if __name__ == '__main__':
    main()
