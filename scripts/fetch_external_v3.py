#!/usr/bin/env python3
"""Pin and verify original publisher audio; never train or score a detector."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import requests

ROOT = Path('runs/external-v3')


def download(url, target, size, algorithm, expected, git_blob=False):
    def checksum(path):
        state = hashlib.new(algorithm)
        if git_blob:
            state.update(f'blob {size}\0'.encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                state.update(block)
        return state.hexdigest()

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == size and checksum(target) == expected:
        return str(target)
    partial = target.with_suffix(target.suffix + '.part')
    with requests.get(url, stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        with partial.open('wb') as stream:
            for block in response.iter_content(1024 * 1024):
                stream.write(block)
    if partial.stat().st_size != size or checksum(partial) != expected:
        raise ValueError(f'Publisher bytes do not match: {target}')
    partial.replace(target)
    print('Verified', target, flush=True)
    return str(target)


def main():
    tree = json.loads((ROOT / 'research/fpv-tree.json').read_text())
    tasks = []
    for entry in tree['tree']:
        if entry['type'] == 'blob':
            tasks.append((f'https://raw.githubusercontent.com/saarmets/fpv-drone-audio-classification-dataset/{tree["sha"]}/{entry["path"]}',
                          ROOT / 'downloads/fpv' / entry['path'], entry['size'], 'sha1', entry['sha'], True))
    metadata = json.loads((ROOT / 'research/itu-zenodo.json').read_text())
    item = metadata['files'][0]
    algorithm, expected = item['checksum'].split(':')
    tasks.append((item['links']['self'], ROOT / 'downloads' / item['key'], item['size'], algorithm, expected, False))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda args: download(*args), tasks))
    archive = ROOT / 'downloads' / item['key']
    target = ROOT / 'downloads/itu'
    with ZipFile(archive) as source:
        for entry in source.infolist():
            destination = (target / entry.filename).resolve()
            if not destination.is_relative_to(target.resolve()):
                raise ValueError('Unsafe publisher archive path')
            if not entry.is_dir():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read(entry))  # ZipFile verifies member CRC.
        print('ITU archive members', len(source.infolist()), flush=True)
        print('ITU metadata paths', [x.filename for x in source.infolist() if not x.filename.endswith('.wav')][:30], flush=True)


if __name__ == '__main__':
    main()
