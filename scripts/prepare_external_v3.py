#!/usr/bin/env python3
"""Audit new publisher audio and reconstruct only sample-verified overlaps."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from drone_detector.common import file_hash, read_json, read_jsonl, write_json, write_jsonl
from drone_detector.data import audit

ROOT = Path('runs/external-v3')
REPORT = Path('reports/external-v3')


def matching_tail(left, right, overlap):
    """Never invent continuity from filenames or class adjacency."""
    return len(left) >= overlap and len(right) >= overlap and np.array_equal(left[-overlap:], right[:overlap])


def prepare_fpv():
    root = ROOT / 'downloads/fpv'
    annotations = []
    for path in sorted(root.rglob('*.wav')):
        relative = str(path.relative_to(root))
        annotations.append({'path': relative, 'id': 'fpv-' + relative.replace('/', '-').removesuffix('.wav'),
            'source': 'saarmets-fpv', 'source_version': read_json(ROOT / 'research/fpv-tree.json')['sha'],
            'original_id': relative, 'session_id': None, 'origin': 'unknown',
            'label': 'non_drone' if relative.startswith('no_drone/') else 'drone',
            'annotation_status': 'weak', 'label_evidence': 'Publisher class folder at pinned revision; no audible boundaries or session documentation supplied.',
            'license': 'CC-BY-4.0; Ralf Saarmets2026',
            'negative_kind': path.stem if relative.startswith('no_drone/') else None})
    assert len(annotations) == 37
    write_jsonl(ROOT / 'fpv-annotations.jsonl', annotations)
    return audit(root, ROOT / 'fpv-inventory.jsonl', ROOT / 'fpv-annotations.jsonl')


def prepare_itu():
    roots = [p.parent for p in (ROOT / 'downloads/itu').rglob('groups.json')]
    if len(roots) != 1:
        raise ValueError('Expected one publisher groups.json in verified archive')
    root = roots[0]
    groups = read_json(root / 'groups.json')
    checksum = root / 'SHA256SUMS'
    checked = 0
    if checksum.is_file():
        for line in checksum.read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            assert file_hash(root / name.lstrip('*')) == expected
            checked += 1
    by_group = defaultdict(list)
    for item in groups['segments']:
        by_group[item['group']].append(item)
    target = ROOT / 'itu-reconstructed'
    target.mkdir(exist_ok=True)
    annotations, joins, pieces = [], [], []
    for group, items in sorted(by_group.items()):
        runs, previous, chunks, originals = [], None, [], []
        for item in items:
            samples, sr = sf.read(root / item['file'], dtype='float32')
            assert sr == 22050 and samples.ndim == 1 and len(samples) == sr
            overlap = sr // 2
            if previous is not None:
                matched = matching_tail(previous, samples, overlap)
                joins.append({'left': originals[-1], 'right': item['file'], 'exact_overlap': matched,
                              'max_abs_overlap_error': float(np.max(np.abs(previous[-overlap:] - samples[:overlap])))})
                if not matched:
                    runs.append((chunks, originals))
                    chunks, originals = [], []
            chunks.append(samples if not chunks else samples[overlap:])
            originals.append(item['file'])
            previous = samples
        if chunks:
            runs.append((chunks, originals))
        for index, (chunks, originals) in enumerate(runs):
            relative = f'{group}-run{index:03d}.wav'
            wave = np.concatenate(chunks)
            sf.write(target / relative, wave, 22050, subtype='FLOAT')
            label = items[0]['label']
            annotations.append({'path': relative, 'id': 'itu-' + relative.removesuffix('.wav'),
                'source': 'itu-aris', 'source_version': 'zenodo22682339-v1.0.0',
                'original_id': group, 'session_id': 'single-continuous-session', 'origin': 'authentic',
                'label': 'drone' if label == 'drone' else 'non_drone', 'annotation_status': 'weak',
                'label_evidence': 'Two-annotator manual drone interval selection, reported kappa0.5868742262; background from unlabelled stretches. Individual disagreements unavailable. Not independently verified temporal audibility.',
                'license': 'CC-BY-4.0; Muhaciroglu and Akgul2026', 'environment': 'outdoor',
                'source_recording': 'One continuous3.12h recording',
                'provenance_review': 'Publisher original capture; sample-exact overlapping exports joined. Entire session is one conservative provenance group.'})
            pieces.append({'id': annotations[-1]['id'], 'publisher_group': group, 'segments': originals,
                           'duration_seconds': len(wave) / 22050})
    write_json(ROOT / 'itu-reconstruction.json', {'archive_checksum_entries_verified': checked, 'joins': joins, 'pieces': pieces,
        'rule': 'Only exactly equal half-second raw PCM overlaps are joined. No missing original-time gaps reconstructed. Background block boundaries retained.'})
    write_jsonl(ROOT / 'itu-annotations.jsonl', annotations)
    report = audit(target, ROOT / 'itu-inventory.jsonl', ROOT / 'itu-annotations.jsonl')
    print('ITU reconstructed', len(pieces), 'sample-matched joins', sum(j['exact_overlap'] for j in joins), '/', len(joins), flush=True)
    return report


def combine():
    rows = []
    for path in [ROOT / 'fpv-inventory.jsonl', ROOT / 'itu-inventory.jsonl']:
        if path.exists():
            rows.extend(read_jsonl(path))
    old = read_jsonl('reports/real-v2/manifest-snapshot.jsonl')
    indices = {key: defaultdict(list) for key in ['sha256', 'decoded_sha256', 'gain_normalized_sha256']}
    for row in old:
        for key, index in indices.items():
            index[row[key]].append(row['id'])
    duplicate_matches = []
    internal = {key: defaultdict(list) for key in indices}
    for row in rows:
        for key, index in indices.items():
            if row[key] in index:
                duplicate_matches.append({'new': row['id'], 'old': index[row[key]], 'kind': key})
            internal[key][row[key]].append(row['id'])
        row['split'] = 'final'
        row['path'] = str((ROOT / row['path']).resolve().relative_to(Path.cwd()))
    # Manifest paths are relative to the project root, explicitly used by the external runner.
    write_jsonl(REPORT / 'recordings.jsonl', rows)
    groups = {source: sorted({r['group_id'] for r in rows if r['source'] == source}) for source in sorted({r['source'] for r in rows})}
    write_json(REPORT / 'data-audit.json', {'recordings': len(rows), 'source_groups': groups,
        'old_recordings_checked': len(old), 'cross_corpus_matches': duplicate_matches,
        'within_source_matches': {key: [ids for ids in index.values() if len(ids) > 1] for key, index in internal.items()},
        'manifest_sha256': file_hash(REPORT / 'recordings.jsonl'),
        'limits': 'Whole-file hashes cannot rule out shifted excerpts, lossy copies or all pretrained-encoder overlap. FPV session/original ancestry remains undocumented; ITU is one original session. No independent audible-event benchmark claimed.'})
    assert not duplicate_matches, 'Resolve overlap before scoring'
    print('Audited', len(rows), 'recordings; exact/decoded/gain-normalized matches to old corpus:', len(duplicate_matches), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', choices=['fpv', 'itu', 'combine'])
    args = parser.parse_args()
    if args.source == 'fpv':
        prepare_fpv()
    elif args.source == 'itu':
        prepare_itu()
    else:
        combine()
