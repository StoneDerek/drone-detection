#!/usr/bin/env python3
"""Export existing numeric models; no training and no assessment access."""

import argparse
import json
import hashlib
import tarfile
import urllib.request
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import firwin

from drone_detector.common import file_hash, read_json, write_json
from drone_detector.model import load_model
from drone_detector.spectrogram import axes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', type=Path, default=Path('reports/spectral-lab-v1/catalog.json'))
    args = parser.parse_args()
    root = Path('web/public/models')
    root.mkdir(parents=True, exist_ok=True)
    catalog = read_json(args.catalog)
    exported = []
    for item in catalog['models']:
        name = item['name']
        model = load_model(args.catalog.parent / item['bundle'], True)
        if model.metadata['schema_version'] == 2 and (model.config.normalize_rms or
                model.config.power_reference != 'context_peak'):
            raise ValueError('Browser spectral frontend supports context-peak models without waveform RMS normalization')
        destination = root / name
        destination.mkdir(exist_ok=True)
        arrays = {}
        for key, array in model.arrays.items():
            array = np.asarray(array, dtype='<f4' if array.dtype.itemsize == 4 else '<f8')
            # Each file remains below the Pages 25 MiB asset limit.
            raw = array.tobytes()
            paths = []
            for offset in range(0, len(raw), 4 * 1024 * 1024):
                path = destination / f'{key}-{len(paths)}.bin'
                path.write_bytes(raw[offset:offset + 4 * 1024 * 1024])
                paths.append(path.name)
            arrays[key] = {'paths': paths, 'dtype': str(array.dtype),
                           'shape': list(array.shape), 'sha256': hashlib.sha256(raw).hexdigest()}
        meta = model.metadata
        info = {'name': name, 'model_id': model.model_id, 'features': meta['features'],
            'family': meta.get('family', meta['features'].get('representation')),
            'decision': meta['decision'], 'arrays': arrays,
            'architecture': meta.get('architecture'), 'gamma': meta.get('gamma'),
            'selection_loss': 'development_consistent_max_balanced_loss' if meta['training'].get('negative_loss', '').startswith('softplus(max') else 'development_balanced_bag_loss',
            'history': meta['training'].get('history', []),
            'selected_epoch': meta['training'].get('selected_epoch'),
            'development': meta['selection']['chosen']['metrics'],
            'training_windows_seen': meta['training'].get('unique_windows_seen_at_selected_epoch'),
            'scope': meta['limitations']}
        if meta['schema_version'] == 2:
            frequency, times, bank = axes(model.config)
            info.update({'shape': [len(frequency), len(times)], 'frequencies': frequency.tolist(),
                         'frame_times': times.tolist()})
            if bank is not None:
                info['bank'] = {'indptr': bank.indptr.tolist(), 'indices': bank.indices.tolist(),
                                'data': bank.data.tolist()}
        write_json(destination / 'model.json', info)
        exported.append({'name': name, 'path': f'models/{name}/model.json',
                         'model_id': model.model_id, 'family': info['family']})
    # Official public TensorFlow.js export, fetched from Google's Kaggle model.
    tarpath = Path('runs/cloudflare-deployment/yamnet-tfjs-v1.tar.gz')
    archive_url = 'https://www.kaggle.com/api/v1/models/google/yamnet/tfJs/tfjs/1/download'
    expected_archive = '604467f7ce0900a27a05f274c7e7a46effbb936788a7cee84648238de5c97c82'
    if not tarpath.exists():
        old_download = tarpath.with_name('yamnet-probe-2.data')
        data = old_download.read_bytes() if old_download.exists() else urllib.request.urlopen(archive_url, timeout=60).read(32 * 1024 * 1024)
        if hashlib.sha256(data).hexdigest() != expected_archive:
            raise ValueError('Official YAMNet TFJS download differs from the pinned archive')
        tarpath.parent.mkdir(parents=True, exist_ok=True)
        tarpath.write_bytes(data)
    if file_hash(tarpath) != expected_archive:
        raise ValueError('Official YAMNet TFJS archive changed')
    with tarfile.open(tarpath) as archive:
        for member in archive.getmembers():
            if member.name not in {'model.json', *[f'group1-shard{i}of4.bin' for i in range(1, 5)]} or not member.isfile():
                raise ValueError('Unexpected official YAMNet archive contents')
            path = root / 'yamnet' / member.name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())
    resamplers = {}
    for rate in [8000, 11025, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 192000]:
        ratio = Fraction(16000, rate)
        up, down = ratio.numerator, ratio.denominator
        half = 10 * max(up, down)
        weights = firwin(2 * half + 1, 1 / max(up, down), window=('kaiser', 5.)).astype(np.float32)
        weights *= up
        prepad = down - half % down
        resamplers[rate] = {'up': up, 'down': down, 'prepad': prepad,
            'remove': (half + prepad) // down, 'weights': weights.astype(float).tolist()}
    write_json(root / 'resamplers.json', resamplers)
    write_json(root / 'catalog.json', {'models': exported, 'defaults': catalog['defaults'],
        'scope': catalog['scope'], 'data_summary': catalog['data_summary'],
        'browser_port': 'Frozen weights and decisions; numerical browser parity checked separately. No retraining.',
        'yamnet_tfjs_source': archive_url,
        'yamnet_tfjs_archive_sha256': file_hash(tarpath)})
    # Test fixtures contain only real DEVELOPMENT audio and Python numeric outputs.
    # They are kept in ignored runs/, never included in the deployed assets.
    print(json.dumps({'models': len(exported), 'max_asset_bytes': max(p.stat().st_size for p in root.rglob('*') if p.is_file())}))


if __name__ == '__main__':
    main()
