#!/usr/bin/env python3
"""Fetch a bounded source-intake plan; preserve and verify completed downloads.

One writer per output directory. A server that ignores ranges requires a full
archive restart after interruption. Completed archives survive later invocations.
First-acquisition hashes are observations, not publisher checksum verification.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import urllib.request
import uuid
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, data):
    payload = json.dumps(data, sort_keys=True, indent=2, allow_nan=False).encode()
    partial = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
    with partial.open('xb') as stream:
        stream.write(payload)
        stream.flush()
    os.replace(partial, path)
    if path.read_bytes() != payload:
        raise ValueError('Receipt read-back failed')


def validate_item(item):
    name, size, checksum = item['filename'], item['bytes'], item['sha256']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name):
        raise ValueError('Unsafe intake filename')
    if not isinstance(size, int) or size <= 0 or not item['url'].startswith('https://'):
        raise ValueError('Invalid intake size/URL')
    if checksum is not None and not re.fullmatch(r'[0-9a-f]{64}', checksum):
        raise ValueError('Invalid expected SHA-256')
    if 'range' in item:
        start, end, source_size = item['range']
        if not all(isinstance(n, int) for n in (start, end, source_size)) or not 0 <= start <= end < source_size or end - start + 1 != size:
            raise ValueError('Invalid source byte range')


def acquire(item, output):
    validate_item(item)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps(item, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    path = output / item['filename']
    receipt = path.with_name(path.name + '.complete.json')
    pending = path.with_name(path.name + '.pending.json')
    existing = next((json.loads(p.read_text()) for p in [receipt, pending] if p.exists()), None)
    if existing and existing['item_identity'] != identity:
        raise ValueError('Intake item changed; use a new intake directory')
    if existing and item['sha256'] is not None and existing['sha256'] != item['sha256']:
        raise ValueError('Receipt hash differs from pinned intake hash')
    expected = item['sha256'] or (existing['sha256'] if existing else None)
    if path.exists():
        if expected is None or path.stat().st_size != item['bytes'] or digest(path) != expected:
            raise ValueError('Existing source is unrecognized or corrupt; preserve it for review')
        result = existing or {'item_identity': identity, 'filename': item['filename'], 'url': item['url'],
            'bytes': item['bytes'], 'sha256': expected,
            'hash_status': 'matched previously pinned SHA-256', 'publisher_checksum_verified': False}
        write_json(receipt, result)
        pending.unlink(missing_ok=True)
        return {**result, 'reused': True}
    partial = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
    headers = {'Accept-Encoding': 'identity'}
    if 'range' in item:
        start, end, source_size = item['range']
        headers['Range'] = f'bytes={start}-{end}'
    request = urllib.request.Request(item['url'], headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if 'range' in item:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{source_size}':
                    raise ValueError('Server did not honor the exact member range; full response refused')
            elif response.status != 200:
                raise ValueError('Full-file transfer did not return HTTP 200')
            count, checksum = 0, hashlib.sha256()
            next_progress = 256 * 1024 * 1024
            print(f'Downloading {item["filename"]}: {item["bytes"] / 1e9:.3f} GB', flush=True)
            with partial.open('xb') as target:
                while True:
                    chunk = response.read(min(1024 * 1024, item['bytes'] - count + 1))
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > item['bytes']:
                        raise ValueError('Source exceeds declared transfer size')
                    checksum.update(chunk)
                    target.write(chunk)
                    if count >= next_progress:
                        print(f'{item["filename"]}: {count / 1e9:.2f}/{item["bytes"] / 1e9:.2f} GB', flush=True)
                        next_progress += 256 * 1024 * 1024
                target.flush()
        observed = checksum.hexdigest()
        if count != item['bytes'] or (expected is not None and expected != observed):
            raise ValueError('Source length/hash differs from intake identity')
        # Verify the stored bytes, not just the network stream's digest.
        if digest(partial) != observed:
            raise ValueError('Source storage read-back failed')
        result = {'item_identity': identity, 'filename': item['filename'], 'url': item['url'],
                  'bytes': count, 'sha256': observed,
                  'hash_status': 'matched previously pinned SHA-256' if expected else 'first acquisition; observed SHA-256 frozen',
                  'publisher_checksum_verified': False}
        write_json(pending, result)
        if path.exists():
            raise ValueError('Another writer created this source; use one active runtime')
        os.replace(partial, path)
        write_json(receipt, result)
        pending.unlink(missing_ok=True)
        return {**result, 'reused': False}
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/source-intake-v2.json'))
    parser.add_argument('--profile', choices=['pilot', 'outdoor'], default='pilot')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-gib', type=float, required=True, help='Declared available project allocation, not a Drive quota measurement')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    items = [row for row in manifest['items'] if args.profile in row['profiles']]
    for item in items:
        validate_item(item)
    total = sum(item['bytes'] for item in items)
    if not items or not args.budget_gib > 0 or total > args.budget_gib * 1024 ** 3:
        parser.error('The selected intake exceeds the declared storage budget')
    args.output.mkdir(parents=True, exist_ok=True)
    # Mounted Drive free-space reporting is not a reliable account-quota query.
    # This only catches an obviously undersized local backing filesystem.
    needed = sum(row['bytes'] for row in items if not (args.output / row['filename']).exists())
    if shutil.disk_usage(args.output).free < needed:
        parser.error('Not enough filesystem space for the pending sources')
    print(f'Intake {args.profile}: {total / 1e9:.3f} GB; no model training', flush=True)
    for item in items:
        result = acquire(item, args.output)
        print(json.dumps({'filename': result['filename'], 'bytes': result['bytes'],
                          'sha256': result['sha256'], 'hash_status': result['hash_status'],
                          'reused': result['reused']}), flush=True)
    print('Sources saved. Corpus quality, labels and split acceptance remain separate gates.', flush=True)


if __name__ == '__main__':
    main()
