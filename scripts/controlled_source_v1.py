#!/usr/bin/env python3
"""Fixed one-microphone transfer check on a separately captured source.

Fetch only channel 0 through exact HTTPS ranges. Verify each original WAV and
JSON against the publisher manifest, then retain lossless PCM-identical FLACs.
The archive checksum is not verified by a partial download. No author code runs.
"""

import argparse
import csv
import gzip
import hashlib
import io
import shutil
import threading
import time
import urllib.error
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.special import expit
from threadpoolctl import threadpool_limits

from cross_condition_v1 import context_starts, fixed_models, source_metrics
from ddl_audio import RangeReader
from drone_detector.audio import decode
from drone_detector.common import canonical, file_hash, read_json, read_jsonl, write_json, write_jsonl
from drone_detector.spectrogram import SpectrogramConfig, spectrogram

ROOT = Path('runs/cross-condition-v1/controlled-source')
REPORT = Path('reports/cross-condition-v1/controlled-source')
LOCAL = threading.local()


class PatientRangeReader(RangeReader):
    """Resume the exact same range after transient server throttling."""

    def read(self, size=-1):
        position = self.position
        for attempt in range(8):
            try:
                return super().read(size)
            except urllib.error.HTTPError as error:
                if error.code not in {429, 502, 503, 504}:
                    raise
                self.position = position
                retry = error.headers.get('Retry-After', '')
                delay = max(60., float(retry) if retry.isdigit() else 60. * (attempt + 1))
                print('SERVER RETRY', error.code, 'waiting', delay, 'seconds', flush=True)
                # This process yields to the parent after launch; the short
                # chunks also make interruption responsive during backoff.
                while delay > 0:
                    pause = min(delay, 30.)
                    time.sleep(pause)
                    delay -= pause
        raise ValueError('Source remains unavailable after server backoff')


def csv_rows(name):
    with (ROOT / 'metadata/dataset/metadata' / f'{name}.csv').open(encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def freeze():
    if (REPORT / 'plan.json').exists():
        raise ValueError('Preserve the existing source plan')
    REPORT.mkdir(parents=True, exist_ok=True)
    records, files = csv_rows('recordings'), csv_rows('files')
    if len(records) != 273 or len(files) != 4368:
        raise ValueError('Publisher inventory changed')
    selected = [row for row in files if row['channel_index'] == '0']
    if len(selected) != 273 or len({r['recording_id'] for r in selected}) != 273:
        raise ValueError('Exactly one prespecified physical microphone per recording')
    lookup = {row['recording_id']: row for row in records}
    members = {row['path']: row for row in read_json(ROOT / 'metadata/BUNDLE_MANIFEST.json')['files']}
    cohort = []
    for row in selected:
        meta = lookup[row['recording_id']]
        path = 'dataset/' + row['relative_path']
        member = members[path]
        if row['sha256'] != member['sha256'] or int(row['file_size_bytes']) != member['bytes']:
            raise ValueError('Publisher indexes disagree')
        json_path = str(Path(path).parent / (row['recording_id'] + '.json'))
        cohort.append({'publisher': meta, 'wav': {'path': path, **member},
                       'json': {'path': json_path, **members[json_path]}, 'sample_count': int(row['sample_count'])})
    inventory = read_json(ROOT / 'inventory.json')
    parent = Path('reports/cross-condition-v1/plan.json')
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(),
            'source': 'https://doi.org/10.5281/zenodo.23613504',
            'archive': {k: inventory[k] for k in ['url', 'size', 'archive_checksum', 'whole_archive_checksum_verified']},
            'publisher_manifest_sha256': file_hash(ROOT / 'metadata/BUNDLE_MANIFEST.json'),
            'parent_panel_sha256': file_hash(parent), 'models': read_json(parent)['models'], 'cohort': cohort,
            'selection': 'All273 main physical recordings, channel0 only:200 hovering DJI Flip,28 field backgrounds,45 controlled replays. Calibration and other channels excluded.',
            'policy': 'Fixed0.5 classifier threshold; any sampled full1s score>=0.5. Hop0.5s, at most60 uniformly spaced full contexts including endpoints. No padding, fitting or threshold selection.',
            'labels': 'Publisher experimental-design presence/absence. Positive event extents are not human-reviewed audible boundaries. Recording-level contrast only; no event, false-alarm/hour or detection-range claim.',
            'grouping': 'Acquisition batch/date, all undocumented field backgrounds together. Not215 independent sessions: many session IDs are per-recording placeholders. One site/device/drone family.',
            'timing': 'This source was inspected after the consumed FPV and fresh Kenya outcomes; all12 earlier model packages and thresholds are carried forward without selecting winners.',
            'storage': 'Original WAV SHA256+ZIP CRC checked in memory; retained FLAC must decode bit-exact int16 PCM. Raw archive MD5 is unverified because only selected members are fetched.'}
    write_json(REPORT / 'plan.json', plan)
    source = REPORT / 'source-records'
    source.mkdir(exist_ok=True)
    for name in ['README.md', 'package_verification.json', 'maintenance_verification_20261109.json']:
        shutil.copyfile(ROOT / name, source / name)
    shutil.copyfile('runs/cross-condition-v1/controlled-source-record.json', source / 'zenodo-record.json')
    shutil.copyfile(ROOT / 'metadata/BUNDLE_MANIFEST.json', source / 'BUNDLE_MANIFEST.json')
    for name in ['recordings', 'files', 'sessions', 'experimental_design']:
        shutil.copyfile(ROOT / 'metadata/dataset/metadata' / f'{name}.csv', source / f'{name}.csv')
    print('FROZEN', len(cohort), 'physical recordings', file_hash(REPORT / 'plan.json'), flush=True)


def remote_zip(plan):
    if not hasattr(LOCAL, 'archive'):
        LOCAL.reader = PatientRangeReader(plan['archive']['url'], plan['archive']['size'], block_size=4 * 1024 * 1024)
        LOCAL.archive = zipfile.ZipFile(LOCAL.reader)
    return LOCAL.archive


def fetch_one(item, plan):
    identifier = item['publisher']['recording_id']
    destination = ROOT / 'audio' / f'{identifier}.flac'
    receipt = ROOT / 'receipts' / f'{identifier}.json'
    if receipt.exists():
        row = read_json(receipt)
        if file_hash(destination) != row['sha256'] or row['original_wav_sha256'] != item['wav']['sha256']:
            raise ValueError('A retained source recording changed')
        return row
    archive = remote_zip(plan)
    raw_json = archive.read(item['json']['path'])
    raw = archive.read(item['wav']['path'])
    for value, expected in [(raw_json, item['json']), (raw, item['wav'])]:
        if len(value) != expected['bytes'] or hashlib.sha256(value).hexdigest() != expected['sha256']:
            raise ValueError('Publisher member SHA256 differs')
    import json
    meta = json.loads(raw_json)
    publisher = item['publisher']
    if meta['identification']['recording_id'] != identifier or bool(meta['target']['present']) != (publisher['has_uav'] == 'true'):
        raise ValueError('Publisher recording target and index disagree')
    pcm, rate = sf.read(io.BytesIO(raw), dtype='int16', always_2d=True)
    info = sf.info(io.BytesIO(raw))
    if rate != 48000 or pcm.shape != (item['sample_count'], 1) or info.subtype != 'PCM_16':
        raise ValueError('Unexpected original PCM format')
    destination.parent.mkdir(exist_ok=True)
    sf.write(destination, pcm, rate, format='FLAC', subtype='PCM_16')
    restored, restored_rate = sf.read(destination, dtype='int16', always_2d=True)
    if restored_rate != rate or not np.array_equal(pcm, restored):
        raise ValueError('Lossless storage changed original PCM')
    metadata_path = REPORT / 'recording-metadata' / f'{identifier}.json'
    metadata_path.parent.mkdir(exist_ok=True)
    metadata_path.write_bytes(raw_json)
    wave, measures = decode(destination)
    quantized = np.rint(wave.astype(np.float64) * 32767).clip(-32768, 32767).astype('<i2')
    peak = float(np.max(np.abs(wave)))
    normalized = np.rint(wave / max(peak, 1e-20) * 1024).astype('<i2')
    origin = publisher['data_origin']
    source = 'nanjing-replay' if origin == 'controlled_replay' else 'nanjing-field'
    row = {'id': identifier, 'path': str(destination), 'sha256': file_hash(destination),
           'original_wav_sha256': item['wav']['sha256'], 'original_json_sha256': item['json']['sha256'],
           'int16_pcm_sha256': hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
           'decoded_sha256': hashlib.sha256(quantized.tobytes()).hexdigest(),
           'gain_normalized_sha256': hashlib.sha256(normalized.tobytes()).hexdigest(),
           'source': source, 'group_id': 'nanjing-' + publisher['acquisition_batch_id'],
           'label': 'drone' if publisher['has_uav'] == 'true' else 'non_drone',
           'origin': 'playback' if origin == 'controlled_replay' else 'authentic',
           'scene_type': publisher['scene_type'], 'publisher_session_id': publisher['session_id'],
           'annotation_status': 'weak', 'label_evidence': 'Publisher experimental-design target presence/absence; no reviewed audible-event interpretation.',
           'channel_index': 0, 'publisher': publisher, 'measures': measures}
    receipt.parent.mkdir(exist_ok=True)
    write_json(receipt, row)
    return row


def fetch():
    plan = read_json(REPORT / 'plan.json')
    rows = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index, row in enumerate(pool.map(lambda item: fetch_one(item, plan), plan['cohort'])):
            rows.append(row)
            if (index + 1) % 20 == 0:
                print('FETCH', index + 1, '/273', round(time.perf_counter() - started, 1), 's', flush=True)
    previous = read_jsonl('reports/real-v2/manifest-snapshot.jsonl') + read_jsonl('reports/external-v3/recordings.jsonl') + read_jsonl('reports/cross-condition-v1/kenya-recordings.jsonl')
    matches = []
    for row in rows:
        for old in previous:
            for field in ['sha256', 'decoded_sha256', 'gain_normalized_sha256']:
                if row[field] == old.get(field):
                    matches.append({'new': row['id'], 'old': old['id'], 'fingerprint': field})
        if any(row['original_wav_sha256'] == old.get('sha256') for old in previous):
            matches.append({'new': row['id'], 'fingerprint': 'original_wav_sha256'})
    write_jsonl(REPORT / 'recordings.jsonl', rows)
    write_json(REPORT / 'data-audit.json', {'plan_sha256': file_hash(REPORT / 'plan.json'),
               'recordings': len(rows), 'retained_bytes': sum(Path(row['path']).stat().st_size for row in rows),
               'seconds': sum(row['measures']['duration_seconds'] for row in rows),
               'conservative_acquisition_groups': len({row['group_id'] for row in rows}),
               'cross_corpus_whole_recording_matches': matches,
               'original_member_sha256_and_crc_checked': True, 'retained_pcm_bit_exact': True,
               'whole_archive_checksum_verified': False,
               'limits': 'Whole-recording fingerprints do not rule out every excerpt or lossy derivative. Capture groups do not represent independent sites or hardware.'})
    if matches:
        raise ValueError('Resolve cross-corpus overlap before scoring')


def evaluate():
    plan = read_json(REPORT / 'plan.json')
    if plan['parent_panel_sha256'] != file_hash('reports/cross-condition-v1/plan.json'):
        raise ValueError('Parent panel changed')
    if read_json(REPORT / 'data-audit.json')['cross_corpus_whole_recording_matches']:
        raise ValueError('Resolve source matches first')
    models = fixed_models(plan)
    rows = read_jsonl(REPORT / 'recordings.jsonl')
    execution = {'plan_sha256': file_hash(REPORT / 'plan.json'), 'runner_sha256': file_hash(__file__),
                 'shared_runner_sha256': file_hash('scripts/cross_condition_v1.py'),
                 'recordings_sha256': file_hash(REPORT / 'recordings.jsonl'),
                 'scope': 'Fresh source at first use, recording-level publisher experimental-design labels; now consumed for future tuning.'}
    ledger = REPORT / 'first-use.json'
    if (REPORT / 'comparison.json').exists():
        raise ValueError('Preserve completed source results')
    if ledger.exists() and read_json(ledger) != execution:
        raise ValueError('Execution changed after first source use')
    write_json(ledger, execution)
    result = {'execution': execution, 'models': {}}
    rules = {}
    for arm in plan['models']:
        rules.setdefault(arm['name'].split('-seed')[0].removesuffix('-logistic'), []).append(arm)
    for rule, arms in rules.items():
        completed_path = REPORT / f'{rule}.json'
        if completed_path.exists():
            result['models'].update(read_json(completed_path)['models'])
            continue
        config = SpectrogramConfig(**arms[0]['features'])
        summaries = {arm['name']: [] for arm in arms}
        with gzip.open(REPORT / f'{rule}.scores.jsonl.gz', 'wb') as stream:
            for index, row in enumerate(rows):
                if file_hash(row['path']) != row['sha256']:
                    raise ValueError('Source changed after audit')
                wave, _ = decode(row['path'])
                starts = context_starts(len(wave), 60)
                if not len(starts):
                    raise ValueError('Unexpected recording too short for a full context')
                images = np.stack([spectrogram(wave[start:start + 16000], config) for start in starts])
                rms = np.array([np.sqrt(np.mean(wave[start:start + 16000].astype(np.float64) ** 2)) for start in starts])
                per_model = {}
                for arm in arms:
                    if arm['features'] != arms[0]['features']:
                        raise ValueError('Shared input rules differ')
                    model = models[arm['name']]
                    if arm['family'] == 'cnn':
                        scores = model.score_images(images)
                    else:
                        features = np.concatenate([images.mean(axis=2), images.std(axis=2)], axis=1).astype(np.float64)
                        scores = expit(((features - model['mean']) / model['scale']) @ model['coef'] + model['intercept'][0])
                    per_model[arm['name']] = scores.astype(float).tolist()
                    summaries[arm['name']].append({key: row[key] for key in ['id', 'source', 'group_id', 'label', 'scene_type']} |
                        {'sampled_contexts': len(starts), 'silent_contexts': int(np.sum(rms < config.silence_rms)),
                         'max_score': float(scores.max()), 'predicted': int(np.any(scores >= .5)),
                         'positive_context_fraction': float(np.mean(scores >= .5))})
                stream.write(canonical({'recording_id': row['id'], 'start_samples': starts.tolist(), 'rms': rms.tolist(), 'scores': per_model}) + b'\n')
                if (index + 1) % 50 == 0:
                    print('EVALUATE', rule, index + 1, '/273', flush=True)
        completed = {'models': {name: {'recordings': values, 'by_source': source_metrics(values)} for name, values in summaries.items()}}
        write_json(completed_path, completed)
        result['models'].update(completed['models'])
        print('COMPLETE', rule, flush=True)
    write_json(REPORT / 'comparison.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['freeze', 'fetch', 'evaluate'])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        {'freeze': freeze, 'fetch': fetch, 'evaluate': evaluate}[args.phase]()
