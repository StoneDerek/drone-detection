#!/usr/bin/env python3
"""Preserve rejected drafts and freeze already-calibrated follow-up metadata."""

import shutil
from pathlib import Path

import numpy as np

from drone_detector.common import file_hash, read_json, revision, write_json
from drone_detector.model import load_model, save_model


def main():
    root = Path('runs/spectral-lab-v1')
    results = read_json(root / 'calibrated-results.json')
    correction_path = root / 'freeze-status-correction.json'
    correction = read_json(correction_path) if correction_path.exists() else {
        'reason': 'Package validation rejected a trained status inherited from follow-up parents after development calibration.',
        'code': revision(), 'weights_decisions_metrics_changed': False, 'models': {}}
    for name, report in results['models'].items():
        source = root / 'calibrated' / name
        model = load_model(source)
        if model.metadata['status'] == 'frozen':
            continue
        if model.metadata['selection']['split'] != 'development' or not model.metadata.get('decision'):
            raise ValueError('Cannot freeze a package without completed development calibration')
        stage = root / 'freeze-correction' / name
        metadata = {**model.metadata, 'status': 'frozen', 'packaging_correction': {
            'code': revision(), 'reason': correction['reason'], 'prior_draft_model_id': model.model_id}}
        fixed = save_model(stage, metadata, model.arrays)
        load_model(stage, require_frozen=True)
        for key in model.arrays:
            assert np.array_equal(fixed.arrays[key], model.arrays[key])
        assert fixed.metadata['decision'] == model.metadata['decision']
        archive = root / 'packaging-errors' / 'unfrozen-calibrated' / name
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(archive))
        shutil.move(str(stage), str(source))
        packaged = Path('artifacts/spectral-lab-v1/models') / name
        if packaged.exists():
            rejected = root / 'packaging-errors' / 'packaged-unfrozen' / name
            rejected.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(packaged), str(rejected))
        correction['models'][name] = {'prior_draft_id': model.model_id, 'frozen_id': fixed.model_id,
            'parameters_sha256': file_hash(source / 'parameters.npz'), 'weights_equal': True, 'decision_equal': True}
        report['prior_draft_model_id'] = report['model_id']
        report['model_id'] = fixed.model_id
    write_json(root / 'calibrated-results.json', results)
    write_json(root / 'freeze-status-correction.json', correction)
    print(correction)


if __name__ == '__main__':
    main()
