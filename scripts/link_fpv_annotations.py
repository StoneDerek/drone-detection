#!/usr/bin/env python3
"""Retain duration-consistent annotation candidates without inventing alignment."""

import argparse
import collections
from pathlib import Path, PurePosixPath

from drone_detector.common import file_hash, read_json, read_jsonl, write_json, write_jsonl
from drone_detector.fpv_lineage import original_intervals, union_length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("work/positive-source-discovery-v1"))
    parser.add_argument("--work", type=Path, default=Path("work/fpv-lineage-v1"))
    args = parser.parse_args()
    metadata = {r['member']: r for r in read_jsonl(args.work / "decoded-members.jsonl")}
    documents = collections.defaultdict(list)
    for item in read_json(args.metadata / "annotation-manifest.json"):
        if item['archive'] != 'Fpv.zip':
            continue
        path = args.metadata / "field-annotations" / item['local_name']
        if file_hash(path) != item['sha256']:
            raise ValueError("Annotation bytes changed")
        intervals = []
        for index, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            a, b, label = line.split(maxsplit=2)
            intervals.append(dict(start=float(a), end=float(b), label=label, row=index))
        for duplicate_offset in [0, 1]:
            counts = collections.Counter()
            for interval in intervals:
                raw = interval['label']
                occurrence = counts[raw]
                counts[raw] += 1
                stem = raw if occurrence == 0 else f"{raw}-{occurrence + duplicate_offset}"
                documents[str(PurePosixPath(item['member']).parent)].append(dict(
                    **interval, stem=stem, document_sha256=item['sha256'],
                    duplicate_suffix_hypothesis=duplicate_offset,
                ))
    outcomes = []
    for original in original_intervals(read_json(args.metadata / "Fpv.zip.directory.json")):
        member = original['path']
        path = PurePosixPath(member)
        view = str(path.parent.parent)
        if member not in metadata:
            outcomes.append(dict(member=member, status='audio_not_decoded', candidates=[]))
            continue
        duration = metadata[member]['duration_seconds']
        tolerance = max(.002, 2 / metadata[member]['sample_rate'])
        named = [r for r in documents.get(view, []) if r['stem'] == path.stem]
        matches = {}
        for row in named:
            if not row['label'].startswith(path.parent.name):
                continue
            if abs(row['end'] - row['start'] - duration) > tolerance:
                continue
            key = row['document_sha256'], row['row']
            if key not in matches:
                matches[key] = {k: v for k, v in row.items() if k != 'duplicate_suffix_hypothesis'}
                matches[key]['supported_numbering_hypotheses'] = []
            matches[key]['supported_numbering_hypotheses'].append(row['duplicate_suffix_hypothesis'])
        candidates = list(matches.values())
        # Candidates are intentionally not promoted to waveform-aligned truth.
        status = ('unique_duration_consistent_candidate' if len(candidates) == 1 else
                  'ambiguous_duration_consistent_candidates' if candidates else
                  'duration_conflict' if named else 'no_matching_released_annotation')
        conflicts = []
        for row in candidates:
            for other in documents.get(view, []):
                if (other['document_sha256'] == row['document_sha256']
                        and not other['label'].startswith(path.parent.name)):
                    a, b = max(row['start'], other['start']), min(row['end'], other['end'])
                    if b > a:
                        conflicts.append((a, b))
        outcomes.append(dict(
            member=member, status=status, native_duration_seconds=duration,
            tolerance_seconds=tolerance, candidates=candidates,
            opposite_label_overlap_seconds=union_length(conflicts),
        ))
    write_jsonl(args.work / "annotation-candidates.jsonl", outcomes)
    summary = dict(
        original_intervals_accounted=len(outcomes),
        status_counts=dict(collections.Counter(r['status'] for r in outcomes)),
        intervals_with_opposing_annotation_overlap=sum(r.get('opposite_label_overlap_seconds', 0) > 0 for r in outcomes),
        scope='Filename/export-numbering and native-duration candidates only. Exact continuous-source waveform alignment and independent audibility remain unverified.',
        detector_scores_generated=0,
    )
    write_json(args.work / "annotation-summary.json", summary)
    print(summary)


if __name__ == '__main__':
    main()
