#!/usr/bin/env python3
"""Recover frozen numeric CNN exports without retraining or replacing originals."""

import hashlib
import json
from pathlib import Path

import numpy as np
import requests

from drone_detector.common import file_hash, read_json, write_json
from drone_detector.model import load_model, save_model

ROOT = Path('runs/external-v3/recovery')
BASE = 'https://drone-audio-lab.pages.dev/'


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    parity = read_json('reports/cloudflare-v3/parity.json')
    for name in ['coherent-frequency-seed42', 'coherent-frequency-seed7']:
        destination = ROOT / name
        destination.mkdir(exist_ok=True)
        response = requests.get(BASE + f'models/{name}/model.json', timeout=30)
        response.raise_for_status()
        (destination / 'browser-model.json').write_bytes(response.content)
        info = response.json()
        evaluation = read_json(f'reports/cnn-foundations-v1/{name}-evaluation.json')
        assert info['model_id'] == evaluation['model_id']
        assert info['decision'] == evaluation['development']['decision']
        assert info['architecture'] == evaluation['architecture']
        assert all(r['models'][name]['model_id'] == info['model_id'] for r in parity['results'])
        arrays = {}
        for key, spec in info['arrays'].items():
            chunks = []
            for path in spec['paths']:
                response = requests.get(BASE + f'models/{name}/{path}', timeout=30)
                response.raise_for_status()
                (destination / path).write_bytes(response.content)
                chunks.append(response.content)
            raw = b''.join(chunks)
            assert hashlib.sha256(raw).hexdigest() == spec['sha256']
            arrays[key] = np.frombuffer(raw, dtype=spec['dtype']).reshape(spec['shape']).copy()
        metadata = {
            'schema_version': 2, 'status': 'frozen', 'family': info['family'],
            'class_mapping': {'0': 'non_drone', '1': 'drone'},
            'features': info['features'], 'architecture': info['architecture'],
            'weight_shapes': [list(arrays[f'w{i}'].shape) for i in range(len(arrays))],
            'decision': info['decision'], 'training': evaluation['training'],
            'recovery': {'source_model_id': info['model_id'], 'source_url': BASE + f'models/{name}/model.json',
                         'browser_metadata_sha256': file_hash(destination / 'browser-model.json'),
                         'array_sha256': {k: v['sha256'] for k, v in info['arrays'].items()},
                         'reason': 'Original numeric CNN bundle is absent from restored Git tree; recover the deployed numeric export. New packaging identity, same declared frozen numerical model.'},
            'limitations': info['scope'],
        }
        bundle = destination / 'bundle'
        if not bundle.exists():
            save_model(bundle, metadata, arrays)
        model = load_model(bundle, require_frozen=True)
        records.append({'name': name, 'bundle': str(bundle), 'source_model_id': info['model_id'],
                        'recovered_model_id': model.model_id, 'arrays_verified': True,
                        'decision_matches_committed_report': True})
        print(json.dumps(records[-1]), flush=True)
    write_json(ROOT / 'recovery.json', records)


if __name__ == '__main__':
    main()
