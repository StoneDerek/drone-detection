#!/usr/bin/env python3
"""Copy the frozen point-microphone index byte-for-byte into a verified cache.

No crops, resampling, gain changes or class scores. Original source archives stay
in Drive. Archive working copies use local SSD; completed point files and their
receipts survive runtime replacement. One active writer per preparation folder.
"""

import argparse
import binascii
import hashlib
import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from audit_native_recording_pairs import native_facts
from fetch_source_intake import digest, write_json

ROOT = Path(__file__).resolve().parents[1]


def output_name(number, row):
    # Use the frozen row position and basename; never extract ZIP directory paths.
    leaf = row['member_path'].replace('\\', '/').rsplit('/', 1)[-1]
    if not leaf.lower().endswith('.wav') or leaf in {'.', '..'} or '\0' in leaf:
        raise ValueError('Invalid point-microphone member name')
    return f'{number:03d}-' + leaf


def existing_point(row, path, identity):
    complete = path.with_name(path.name + '.complete.json')
    pending = path.with_name(path.name + '.pending.json')
    receipt = next((json.loads(p.read_text()) for p in [complete, pending] if p.exists()), None)
    if not path.exists():
        return None
    if receipt is None or receipt['run_identity'] != identity or receipt['source_archive_sha256'] != row['source_archive_sha256'] or receipt['member_path'] != row['member_path']:
        raise ValueError('Unrecognized existing point file; preserve it for review')
    if path.stat().st_size != row['member_bytes'] or digest(path) != receipt['sha256']:
        raise ValueError('Completed point cache changed; do not silently replace it')
    payload = path.read_bytes()
    if binascii.crc32(payload) != row['member_crc32'] or (row['member_sha256'] is not None and receipt['sha256'] != row['member_sha256']):
        raise ValueError('Cached point differs from the frozen member integrity record')
    write_json(complete, receipt)
    pending.unlink(missing_ok=True)
    return receipt


def copy_point(row, archive, path, identity):
    info = archive.getinfo(row['member_path'])
    if info.file_size != row['member_bytes'] or info.CRC != row['member_crc32'] or info.file_size > 256 * 1024 * 1024:
        raise ValueError('Point member metadata differs from the frozen index')
    payload = archive.read(info)  # Verify the complete ZIP member CRC.
    checksum = hashlib.sha256(payload).hexdigest()
    if row['member_sha256'] is not None and checksum != row['member_sha256']:
        raise ValueError('Point member differs from its previously observed SHA')
    facts, _ = native_facts(payload)
    if (facts['channels'] != 1 or facts['subtype'] != 'FLOAT'
            or facts['sample_rate'] != row['sample_rate'] or facts['frames'] != row['native_frames']):
        raise ValueError('Native point format/frame count differs; no implicit conversion allowed')
    receipt = {'run_identity': identity, 'member_path': row['member_path'],
               'source_archive_sha256': row['source_archive_sha256'], 'bytes': len(payload),
               'sha256': checksum, 'crc32': info.CRC, 'complete_member_crc_verified': True,
               'native_bytes_unchanged': True, 'audio': facts}
    partial = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
    pending = path.with_name(path.name + '.pending.json')
    complete = path.with_name(path.name + '.complete.json')
    try:
        with partial.open('xb') as stream:
            stream.write(payload)
            stream.flush()
        if digest(partial) != checksum:
            raise ValueError('Point file storage read-back failed')
        write_json(pending, receipt)
        if path.exists():
            raise ValueError('Another writer created this point file')
        os.replace(partial, path)
        write_json(complete, receipt)
        pending.unlink(missing_ok=True)
        return receipt
    finally:
        partial.unlink(missing_ok=True)


def stage_archive(source, cache, locked):
    if source.stat().st_size != locked['bytes']:
        raise ValueError('Source archive byte length changed')
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (locked['sha256'][:12] + '-' + locked['filename'])
    if target.exists():
        if target.stat().st_size != locked['bytes'] or digest(target) != locked['sha256']:
            raise ValueError('Local source working copy changed')
        return target
    for old in cache.glob(target.name + '.partial-*'):
        old.unlink()  # Incomplete copies in this dedicated local cache only.
    if shutil.disk_usage(cache).free < locked['bytes'] + 64 * 1024 * 1024:
        raise ValueError('Local SSD needs space for one source archive plus working memory margin')
    partial = target.with_name(target.name + '.partial-' + uuid.uuid4().hex)
    checksum, count = hashlib.sha256(), 0
    next_progress = 512 * 1024 * 1024
    try:
        with source.open('rb') as incoming, partial.open('xb') as output:
            while chunk := incoming.read(1024 * 1024):
                count += len(chunk)
                if count > locked['bytes']:
                    raise ValueError('Source archive grew during staging')
                checksum.update(chunk)
                output.write(chunk)
                if count >= next_progress:
                    print(f'{locked["filename"]}: staged {count / 1e9:.2f}/{locked["bytes"] / 1e9:.2f} GB', flush=True)
                    next_progress += 512 * 1024 * 1024
        if count != locked['bytes'] or checksum.hexdigest() != locked['sha256'] or digest(partial) != locked['sha256']:
            raise ValueError('Source archive does not match the source lock/storage read-back')
        os.replace(partial, target)
        return target
    finally:
        partial.unlink(missing_ok=True)


def prepare(sources, output, cache, index_path, lock_path, budget_gib):
    sources, output, cache = sources.resolve(), output.resolve(), cache.resolve()
    if (output == sources or output.is_relative_to(sources) or sources.is_relative_to(output)
            or cache == sources or cache.is_relative_to(sources)):
        raise ValueError('Preparation output/cache must be separate from original source storage')
    index = [json.loads(line) for line in index_path.read_text().splitlines()]
    locked = {row['filename']: row for row in json.loads(lock_path.read_text())['sources']}
    if not index or any(row['input_kind'] != 'point_microphone'
                        or not row['member_path'].endswith(('-Mic1.wav', '_BSWA.wav')) for row in index):
        raise ValueError('The preparation index must contain only individual point microphones')
    seen = set()
    for row in index:
        key = (row['source_archive'], row['member_path'])
        if key in seen or row['source_archive_sha256'] != locked[row['source_archive']]['sha256']:
            raise ValueError('Duplicate member or changed source reference')
        seen.add(key)
    total = sum(row['member_bytes'] for row in index)
    if not budget_gib > 0 or total > budget_gib * 1024 ** 3:
        raise ValueError('Point-cache plan exceeds the declared additional storage allocation')
    identity_data = {'schema_version': 1, 'index_sha256': digest(index_path), 'source_lock_sha256': digest(lock_path),
                     'preparer_sha256': digest(__file__),
                     'native_qa_sha256': digest(ROOT / 'scripts/audit_native_recording_pairs.py'),
                     'intake_io_sha256': digest(ROOT / 'scripts/fetch_source_intake.py'),
                     'environment_lock_sha256': digest(ROOT / 'uv.lock'),
                     'numpy': np.__version__, 'soundfile': sf.__version__, 'libsndfile': sf.__libsndfile_version__,
                     'policy': 'Exact original point WAV bytes; finite mono FLOAT frames; no labels inferred'}
    identity = hashlib.sha256(json.dumps(identity_data, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / 'identity.json'
    if marker.exists() and json.loads(marker.read_text()) != identity_data:
        raise ValueError('Preparation identity changed; preserve this run and use a new output folder')
    if not marker.exists():
        write_json(marker, identity_data)
    point_dir = output / 'native-point'
    point_dir.mkdir(exist_ok=True)
    results = {}
    for archive_name in sorted({row['source_archive'] for row in index}):
        members = [(number, row) for number, row in enumerate(index, 1) if row['source_archive'] == archive_name]
        missing = []
        for number, row in members:
            path = point_dir / output_name(number, row)
            done = existing_point(row, path, identity)
            if done is None:
                missing.append((number, row, path))
            else:
                results[number] = done
        if missing:
            print('Verifying/staging source:', archive_name, flush=True)
            local = stage_archive(sources / archive_name, cache, locked[archive_name])
            with zipfile.ZipFile(local) as archive:
                for number, row, path in missing:
                    # Old partials are incomplete copies from this owned run;
                    # originals and completed sources remain untouched.
                    for old in point_dir.glob(path.name + '.partial-*'):
                        old.unlink()
                    results[number] = copy_point(row, archive, path, identity)
                    print(f'Prepared {number}/{len(index)}: {row["member_path"]}', flush=True)
            local.unlink()  # Only our verified local working copy, not Drive input.
    rows = [{**row, 'path': 'native-point/' + output_name(number, row),
             'sha256': results[number]['sha256'], 'quality': results[number]['audio'],
             'complete_member_crc_verified_here': True, 'native_bytes_unchanged': True}
            for number, row in enumerate(index, 1)]
    payload = ''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in rows)
    temporary = output / ('recordings.partial-' + uuid.uuid4().hex)
    temporary.write_text(payload)
    os.replace(temporary, output / 'recordings.jsonl')
    if (output / 'recordings.jsonl').read_text() != payload:
        raise ValueError('Prepared index storage read-back failed')
    summary = {'completed_point_files': len(rows), 'bytes': total,
               'point_file_hours': sum(row['duration_seconds'] for row in rows) / 3600,
               'native_bytes_unchanged': True, 'all_completed_member_crcs_checked': True,
               'all_completed_files_finite': True, 'source_archive_reference_count': len({row['source_archive'] for row in rows}),
               'labels_accepted_for_training_or_test': False, 'new_drone_absent_hours_established': 0,
               'scope': 'Source-file preparation only; no new independent recordings, event labels or model evidence'}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--budget-gib', type=float, required=True)
    parser.add_argument('--index', type=Path, default=ROOT / 'data/outdoor-point-recordings-v1.jsonl')
    parser.add_argument('--source-lock', type=Path, default=ROOT / 'data/source-lock-v1.json')
    args = parser.parse_args()
    prepare(args.sources, args.output, args.cache, args.index, args.source_lock, args.budget_gib)


if __name__ == '__main__':
    main()
