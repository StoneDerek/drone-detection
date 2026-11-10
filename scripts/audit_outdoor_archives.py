#!/usr/bin/env python3
"""Inventory AuDroK outdoor ZIPs locally or through strictly verified ranges.

Remote URLs belong in an untracked input file. Reports contain publisher source
URLs and receipt hashes, never private Drive file IDs or confirmation URLs.
"""

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

from audit_dronenoise_headers import wav_header
from ddl_audio import RangeReader
from drone_detector.source_inventory import outdoor_name, summarize_views, validate_completion_receipt

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


class PacedReader(RangeReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_network_read = 0.

    def read(self, size=-1):
        end = min(self.length, self.position + (size if size >= 0 else self.length))
        first, last = self.position // self.block_size, (end - 1) // self.block_size
        cold = any(index not in self.cache and not (self.cache_dir / f'{index}.bin').exists()
                   for index in range(first, last + 1))
        if cold:
            time.sleep(max(0., 1. - (time.monotonic() - self.last_network_read)))
        before = self.transferred
        payload = super().read(size)
        if before != self.transferred:
            self.last_network_read = time.monotonic()
        return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intake', type=Path, help='Local source-archives folder')
    parser.add_argument('--remote-config', type=Path, help='Untracked filename -> range-download URL map')
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--archive', choices=['Renamed_Dresden-II-S-A.zip', 'Renamed_Dresden-II-S-B.zip'])
    parser.add_argument('--block-size', type=int, default=65536)
    args = parser.parse_args()
    if (args.intake is None) == (args.remote_config is None):
        parser.error('Choose exactly one of --intake or --remote-config')
    if not 4096 <= args.block_size <= 8 * 1024 * 1024:
        parser.error('Range block size must be between4KiB and8MiB')
    args.output.mkdir(parents=True, exist_ok=False)
    urls = json.loads(args.remote_config.read_text()) if args.remote_config else {}
    plan = json.loads((ROOT / 'data/source-intake-v2.json').read_text())
    items = [row for row in plan['items'] if row['id'].startswith('audrok-outdoor-')
             and (args.archive is None or row['filename'] == args.archive)]
    facts, archives = [], []
    for item in items:
        name = item['filename']
        receipt = json.loads((args.receipts / (name + '.complete.json')).read_text())
        validate_completion_receipt(item, receipt)
        reader = None
        if args.intake:
            source = args.intake / name
            if source.stat().st_size != item['bytes'] or digest(source) != receipt['sha256']:
                raise ValueError('Local archive differs from its completed receipt')
        else:
            suffix = '.blocks' if args.block_size == 65536 else f'.blocks-{args.block_size}'
            reader = PacedReader(urls[name], item['bytes'], block_size=args.block_size,
                                 cache_dir=args.cache / (name + suffix))
            source = reader
        archive_rows = []
        with zipfile.ZipFile(source) as archive:
            entries = [info for info in archive.infolist() if not info.is_dir()]
            for number, info in enumerate(entries, 1):
                if not info.filename.lower().endswith('.wav'):
                    raise ValueError('Unexpected non-WAV member; inspect before proceeding')
                with archive.open(info) as member:
                    prefix = member.read(65536)
                header = wav_header(prefix, info.file_size)
                row = {'archive': name, 'path': info.filename, 'bytes': info.file_size,
                       'compressed_bytes': info.compress_size, 'crc32': info.CRC,
                       'prefix_sha256': hashlib.sha256(prefix).hexdigest(),
                       'whole_member_crc_verified': False, **outdoor_name(info.filename), **header}
                archive_rows.append(row)
                if number % 16 == 0 or number == len(entries):
                    print(name, number, '/', len(entries), 'headers inspected', flush=True)
        facts.extend(archive_rows)
        (args.output / (name + '.headers.jsonl')).write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in archive_rows))
        archives.append({'filename': name, 'source_url': item['url'], 'bytes': item['bytes'],
                         'sha256_reference': receipt['sha256'],
                         'completion_receipt_checked': True,
                         'archive_sha256_independently_rehashed': args.intake is not None,
                         'new_network_bytes': reader.transferred if reader else 0,
                         'wav_members': len(archive_rows)})
    result = {'schema_version': 1, 'archives': archives, **summarize_views(facts),
              'whole_members_crc_verified': False, 'accepted_training_or_test_hours': None,
              'split': None, 'detector_scores_generated': False}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ['archives', 'measurements']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
