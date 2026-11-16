#!/usr/bin/env python3
"""Compare complete FPV member bytes with recorded prior source/file SHA-256s."""

import argparse
import collections
from pathlib import Path

from drone_detector.common import file_hash, read_jsonl, write_json, write_jsonl
from drone_detector.fpv_lineage import identifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('work/fpv-lineage-v1'))
    parser.add_argument('--manifest', type=Path, action='append', required=True)
    args = parser.parse_args()
    index, manifests = collections.defaultdict(list), []
    for path in args.manifest:
        rows = read_jsonl(path)
        counts = collections.Counter()
        manifest_sha = file_hash(path)
        for row in rows:
            for kind, value in [('stored_file', row.get('sha256')),
                                ('native_member', row.get('native', {}).get('member_sha256'))]:
                if value:
                    counts[kind] += 1
                    index[value].append(dict(manifest_sha256=manifest_sha, hash_kind=kind,
                                             prior_id_hash=identifier(row['id'])))
        manifests.append(dict(manifest_sha256=manifest_sha, rows=len(rows),
                              available_byte_hash_fields=dict(counts),
                              source_counts=dict(collections.Counter(r['source'] for r in rows))))
    candidates = read_jsonl(args.work / 'decoded-members.jsonl') + read_jsonl(args.work / 'raw-members.jsonl')
    matches = [dict(member_id=identifier(r['member']), sha256=r['sha256'], matches=index[r['sha256']])
               for r in candidates if r['sha256'] in index]
    write_jsonl(args.work / 'prior-byte-identity-matches.jsonl', matches)
    result = dict(
        candidate_audio_members_compared=len(candidates), prior_manifest_rows_summed=sum(r['rows'] for r in manifests),
        candidate_members_with_recorded_byte_identity=len(matches), prior_manifests=manifests,
        scope='Exact whole-file SHA-256 against recorded stored-file/native-member identities only. Manifest rows can overlap. Different crops, containers, encodings, gains and shifted excerpts are not excluded by a zero result. Historical manifests without native hashes cannot certify raw-source coverage.',
        detector_scores_generated=0,
    )
    write_json(args.work / 'prior-byte-identity-summary.json', result)
    print(result)


if __name__ == '__main__':
    main()
