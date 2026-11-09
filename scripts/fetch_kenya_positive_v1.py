#!/usr/bin/env python3
"""Recover exactly the prespecified human-primary-label positives, before scores."""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import requests

from drone_detector.common import file_hash, read_json, read_jsonl, write_json, write_jsonl
from drone_detector.data import audit

ROOT = Path('runs/cross-condition-v1/kenya')
REPORT = Path('reports/cross-condition-v1')


def primary_drone_labels(path):
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(path) as archive:
        strings = [''.join(element.itertext()) for element in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si', ns)]
        result = []
        # The pinned workbook's sheet2 is the Human-Annotated tab. Verify its
        # header rather than treating a comment in a later column as the label.
        rows = ET.fromstring(archive.read('xl/worksheets/sheet2.xml')).findall('s:sheetData/s:row', ns)
        header = None
        for row in rows:
            cells = {}
            for cell in row.findall('s:c', ns):
                column = ''.join(c for c in cell.attrib['r'] if c.isalpha())
                value = cell.find('s:v', ns)
                cells[column] = (strings[int(value.text)] if cell.attrib.get('t') == 's' else value.text) if value is not None else ''
            if header is None:
                header = cells
                if cells.get('A') != 'Filename' or cells.get('C') != 'Label':
                    raise ValueError('Publisher label columns changed')
            elif cells.get('C', '').strip().lower() == 'drone':
                result.append(cells['A'])
        return sorted(result)


def main():
    source = read_json(REPORT / 'source-freeze.json')
    if source['parent_plan_sha256'] != file_hash(REPORT / 'plan.json'):
        raise ValueError('Plan changed after source freeze')
    ROOT.mkdir(parents=True, exist_ok=True)
    for item in source['files']:
        path = ROOT / item['name']
        if not path.exists():
            url = f'https://raw.githubusercontent.com/{source["repository"]}/{source["revision"]}/{item["name"]}'
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            path.write_bytes(response.content)
        raw = path.read_bytes()
        git_hash = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        if len(raw) != item['size'] or git_hash != item['git_blob_sha1']:
            raise ValueError('Pinned publisher Git bytes differ')
    names = primary_drone_labels(ROOT / 'Dataset_Description.xlsx')
    expected = sorted(read_json(REPORT / 'plan.json')['fresh_positive_check']['selection'])
    if names != expected:
        raise ValueError('The pinned primary human labels differ from the prespecified cohort')
    annotations = [{'path': name, 'id': 'kenya-' + name.removesuffix('.wav'),
        'source': 'kenya-forest-human-drone', 'source_version': source['revision'],
        'session_id': 'unknown-capture-date', 'original_id': 'ZOOM0071', 'origin': 'authentic',
        'label': 'drone', 'annotation_status': 'weak',
        'label_evidence': 'Human-Annotated spreadsheet primary column C is Drone. Review concerns dominant clip sound; no exact audible boundaries or reviewed negative intervals.',
        'license': 'CC-BY-4.0; Kenya Forest Sounds Dataset authors, Zenodo21443074',
        'source_recording': 'ZOOM0071 apparent original;five adjacent excerpt names, one conservative component',
        'provenance_review': 'Publisher field collection; original Git blobs pinned, apparent original retained as one group. No independent-flight or raw-timeline claim.'}
        for name in names]
    write_jsonl(ROOT / 'annotations.jsonl', annotations)
    inventory = ROOT / 'inventory.jsonl'
    if not inventory.exists():
        audit(ROOT, inventory, ROOT / 'annotations.jsonl')
    rows = read_jsonl(inventory)
    old = read_jsonl('reports/real-v2/manifest-snapshot.jsonl') + read_jsonl('reports/external-v3/recordings.jsonl')
    matches = []
    for row in rows:
        for previous in old:
            for key in ['sha256', 'decoded_sha256', 'gain_normalized_sha256']:
                if row[key] == previous[key]:
                    matches.append({'new': row['id'], 'old': previous['id'], 'fingerprint': key})
        row['path'] = str((ROOT / row['path']).resolve().relative_to(Path.cwd()))
        row['split'] = 'final'
    write_jsonl(REPORT / 'kenya-recordings.jsonl', rows)
    summary = {'source_sha256': file_hash(REPORT / 'source-freeze.json'), 'human_primary_positive_files': len(rows),
               'conservative_original_groups': len({row['group_id'] for row in rows}),
               'seconds': sum(row['duration_seconds'] for row in rows),
               'cross_corpus_whole_file_matches': matches,
               'labels': 'Human primary Drone clip labels only; no negative labels, no temporal audibility truth.',
               'limits': 'No exact-file match does not rule out every shifted or lossy derivative. Five adjacent excerpts do not constitute five independent captures.'}
    write_json(REPORT / 'kenya-data-audit.json', summary)
    if matches:
        raise ValueError('Resolve provenance matches before scoring')
    print(summary)


if __name__ == '__main__':
    main()
