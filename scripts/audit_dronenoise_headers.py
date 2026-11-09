#!/usr/bin/env python3
"""Read bounded WAV headers, without downloading full recordings or assigning labels.

Input is a pinned Figshare metadata JSON. Filename-derived event groups are
candidates for provenance review, not verified independent flights/timelines.
"""

import argparse
import concurrent.futures
import hashlib
import json
import re
import struct
import urllib.request
from pathlib import Path


def wav_header(data, file_size):
    if data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        raise ValueError('Only little-endian RIFF/WAVE headers are supported')
    cursor, format_info = 12, None
    while cursor + 8 <= len(data):
        tag, size = struct.unpack_from('<4sI', data, cursor)
        cursor += 8
        if tag == b'fmt ':
            if size < 16 or cursor + size > len(data):
                raise ValueError('Incomplete fmt chunk')
            encoding, channels, rate, byte_rate, block, bits = struct.unpack_from('<HHIIHH', data, cursor)
            if (encoding not in (1, 3) or channels < 1 or rate < 1 or bits < 1
                    or block != channels * ((bits + 7) // 8) or byte_rate != rate * block):
                raise ValueError('Unsupported/inconsistent WAV format')
            format_info = {'encoding': encoding, 'channels': channels, 'sample_rate': rate,
                           'bits_per_sample': bits, 'byte_rate': byte_rate, 'block_align': block}
        if tag == b'data':
            if format_info is None or cursor + size > file_size or size % format_info['block_align']:
                raise ValueError('Invalid data extent')
            return {**format_info, 'data_bytes': size, 'data_offset': cursor,
                    'duration_seconds': size / format_info['byte_rate']}
        cursor += size + (size % 2)
    raise ValueError('Data header not present in bounded prefix')


def audit_member(member, header_dir):
    size = min(4096, member['size'])
    request = urllib.request.Request(member['download_url'], headers={
        'Range': f'bytes=0-{size - 1}', 'Accept-Encoding': 'identity'})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 206 or response.headers.get('Content-Range') != f'bytes 0-{size - 1}/{member["size"]}':
            raise ValueError('Server did not honor the exact bounded range')
        payload = response.read(size + 1)
    if len(payload) != size:
        raise ValueError('Wrong prefix length')
    path = header_dir / f'{member["id"]}.bin'
    path.write_bytes(payload)
    name = member['name']
    return {'file_id': member['id'], 'name': name, 'file_bytes': member['size'],
            'header_bytes': size, 'header_sha256': hashlib.sha256(payload).hexdigest(),
            'calibration_by_name': name.startswith('Calib_'),
            'candidate_event_key': re.sub(r'_M\d+(?=\.wav$)', '', name) if name.startswith('Ed_') else None,
            'grouping_status': 'Filename heuristic only; unresolved provenance',
            **wav_header(payload, member['size'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    # Refuse to overwrite an earlier audit, including an incomplete attempt.
    args.output.mkdir(parents=True, exist_ok=False)
    headers = args.output / 'headers'
    headers.mkdir()
    metadata = json.loads(args.metadata.read_text())
    members = [row for row in metadata['files'] if row['name'].lower().endswith('.wav')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = sorted(pool.map(lambda row: audit_member(row, headers), members), key=lambda row: row['name'])
    (args.output / 'headers.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    groups = {}
    for row in rows:
        if row['candidate_event_key']:
            groups.setdefault(row['candidate_event_key'], []).append(row['duration_seconds'])
    summary = {'scope': 'Header inventory only; no waveform decoding, MD5 validation, labels or detector scoring',
        'metadata_sha256': hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        'wav_files': len(rows), 'header_payload_bytes': sum(row['header_bytes'] for row in rows),
        'calibration_files_by_name': sum(row['calibration_by_name'] for row in rows),
        'noncalibration_file_seconds': sum(row['duration_seconds'] for row in rows if not row['calibration_by_name']),
        'filename_candidate_event_groups': len(groups),
        'sum_of_max_duration_per_candidate_group_seconds': sum(max(values) for values in groups.values()),
        'verified_unique_timeline_hours': None,
        'grouping_warning': 'Candidate grouping does not establish timing, independence or non-overlap; do not count as accepted hours'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
