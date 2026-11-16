#!/usr/bin/env python3
"""Publish aggregate FPV qualification evidence without member filenames."""

import argparse
import collections
import io
import zipfile
from pathlib import Path, PurePosixPath

from drone_detector.common import file_hash, read_json, read_jsonl, versions, write_json
from drone_detector.fpv_lineage import identifier, original_intervals


def distinct(rows, key='member'):
    names = [r[key] for r in rows]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate ledger rows')
    return set(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('work/fpv-lineage-v1'))
    parser.add_argument('--metadata', type=Path, default=Path('work/positive-source-discovery-v1'))
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('reports/fpv-lineage-v1/audit.json'))
    args = parser.parse_args()
    work = args.work
    summary = read_json(work / 'summary.json')
    if file_hash(args.archive) != summary['archive']['sha256']:
        raise ValueError('Verified archive changed')
    entries = read_json(args.metadata / 'Fpv.zip.directory.json')
    originals = {r['path'] for r in original_intervals(entries)}
    lineage = read_jsonl(work / 'lineage.jsonl')
    exports = distinct(lineage)
    decoded = read_jsonl(work / 'decoded-members.jsonl')
    raw = read_jsonl(work / 'raw-members.jsonl')
    expected_raw = {r['path'] for r in entries if '/labeled_data/' not in r['path']
                    and PurePosixPath(r['path']).suffix.lower() in {'.wav', '.m4a'}}
    outer_audio = {r['path'] for r in entries
                   if PurePosixPath(r['path']).suffix.lower() in {'.wav', '.m4a'}}
    if originals | exports | expected_raw != outer_audio:
        raise ValueError('Unaccounted outer audio member')
    if distinct(decoded) != originals | exports or distinct(raw) != expected_raw:
        raise ValueError('Incomplete decoded member accounting')
    if distinct(read_jsonl(work / 'direct-export-to-original.jsonl')) != exports:
        raise ValueError('Incomplete direct lineage accounting')
    for ledger in ['annotation-candidates.jsonl', 'interval-to-raw.jsonl']:
        if distinct(read_jsonl(work / ledger)) != originals:
            raise ValueError('Incomplete original interval accounting')
    if read_json(work / 'decode-failures.json'):
        raise ValueError('Decode failures must be reported explicitly')
    nested = []
    with zipfile.ZipFile(args.archive) as archive:
        for item in archive.infolist():
            if item.filename.lower().endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(archive.read(item))) as inner:
                    nested.append(dict(member_id=identifier(item.filename),
                                       suffix_counts=dict(collections.Counter(
                                           PurePosixPath(r.filename).suffix.lower()
                                           for r in inner.infolist() if not r.is_dir()))))
    capture_rows = collections.defaultdict(list)
    for row in decoded:
        if row['member'] in originals:
            key = '/'.join(PurePosixPath(row['member']).parts[:3])
            capture_rows[key].append(row)
    captures = []
    for key, rows in sorted(capture_rows.items()):
        captures.append(dict(
            capture_folder_id=identifier(key), labelled_interval_members=len(rows),
            publisher_class_counts=dict(collections.Counter(PurePosixPath(r['member']).parent.name for r in rows)),
            labelled_view_folder_count=len({str(PurePosixPath(r['member']).parent.parent) for r in rows}),
            publisher_partitions=sorted({r['partition'] for r in lineage if key in r['candidate_capture_groups']}),
        ))
    code = ['src/drone_detector/fpv_lineage.py', 'src/drone_detector/corpus_intake.py',
            'src/drone_detector/common.py', 'tests/test_fpv_lineage.py']
    code += ['scripts/' + name + '.py' for name in [
        'prepare_fpv_lineage_inputs', 'audit_fpv_lineage', 'verify_fpv_direct_lineage',
        'audit_fpv_split_overlap', 'link_fpv_annotations', 'audit_fpv_raw_links',
        'audit_fpv_prior_identity', 'summarize_fpv_lineage']]
    ledger_names = [
        'summary.json', 'lineage.jsonl', 'decoded-members.jsonl', 'decode-failures.json',
        'partition-to-original.jsonl', 'export-to-partition.jsonl',
        'direct-export-to-original.jsonl', 'direct-lineage-summary.json',
        'verified-split-overlap-summary.json', 'verified-split-overlaps.json',
        'annotation-candidates.jsonl', 'annotation-summary.json',
        'raw-members.jsonl', 'interval-to-raw.jsonl', 'raw-links-summary.json',
        'prior-byte-identity-matches.jsonl', 'prior-byte-identity-summary.json',
        'acquisition-complete.json', 'waveform-audit-complete.json',
    ]
    result = dict(
        scope='Complete source qualification. No classifier inference, training change, or confirmation result.',
        source_url='https://zenodo.org/records/21935566', source_license='CC BY 4.0',
        publisher_creator='-, - (authorship is not independently established)',
        archive=summary['archive'], declared_lineage=summary['metadata'],
        labelled_and_export_waveforms=summary['waveforms'],
        direct_lineage=read_json(work / 'direct-lineage-summary.json'),
        verified_split_overlap=read_json(work / 'verified-split-overlap-summary.json'),
        annotation_candidates=read_json(work / 'annotation-summary.json'),
        raw_links=read_json(work / 'raw-links-summary.json'),
        prior_byte_identity=read_json(work / 'prior-byte-identity-summary.json'),
        complete_audio_inventory=dict(
            labelled_interval_members=len(originals), publisher_export_members=len(exports),
            other_raw_audio_members=len(raw), total_outer_audio_members_decoded=len(decoded) + len(raw),
            decode_failures=0, nested_archive_directories=nested,
            native_format_counts=dict(collections.Counter(
                f"{r['sample_rate']}Hz/{r['channels']}ch/{r['subtype']}" for r in decoded + raw)),
        ),
        capture_folder_groups=captures,
        group_scope='Conservative publisher folder keys, not independently verified recording campaigns. Parallel device folders do not create independent trials.',
        versions=versions(), detector_scores_generated=0,
        code_sha256={name: file_hash(Path(__file__).resolve().parents[1] / name) for name in code},
        private_ledger_sha256={name: file_hash(work / name) for name in ledger_names},
        input_metadata_sha256={name: file_hash(args.metadata / name)
                               for name in ['Fpv.zip.directory.json', 'annotation-manifest.json']},
        remaining_checks=[
            'Project-wide duplicate/excerpt comparison; neither source text search nor exact within-archive matching proves freshness.',
            'Annotation-to-continuous-audio alignment, capture/view synchronization and independent label/audibility review.',
            'Publisher authorship/capture provenance and disposition of unmatched or unprovided declared parents.',
        ],
        disposition='Unscored candidate. Preserve all source evidence; quarantine unresolved derived exports. Do not adopt publisher splits as an unfamiliar-capture test or add this source to the running experiment.',
    )
    write_json(args.output, result)
    print(args.output)


if __name__ == '__main__':
    main()
