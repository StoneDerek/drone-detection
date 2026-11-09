"""Numerical/input-leakage guards, not acoustic reliability evidence."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from drone_detector.common import file_hash


def script(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_one_second_descriptors_match_independent_unpadded_fft_reference():
    task = script('human_label_task_v1')
    audio = (.07 * np.sin(2 * np.pi * 420 * np.arange(16000) / 16000)
             + .02 * np.cos(2 * np.pi * 1610 * np.arange(16000) / 16000)).astype(np.float32)
    got = task.one_second_features(audio)
    # Independent direct FFT uses exactly47 complete frames; no synthetic tail.
    frames = np.stack([audio[start:start + 4096] for start in range(0, 16000 - 4096 + 1, 256)])
    hann = .5 - .5 * np.cos(2 * np.pi * np.arange(4096) / 4096)
    power = np.abs(np.fft.rfft(frames * hann, axis=1)).T ** 2
    _, _, bank = task.axes(task.CONFIG)
    bands = bank @ power
    image = np.clip(1 + 10 * np.log10(np.maximum(bands / bands.max(), 1e-30)) / 80, 0, 1)
    expected = np.concatenate([image.mean(axis=1), image.std(axis=1)])
    np.testing.assert_allclose(got, expected, atol=2e-4)
    assert got.shape == (512,)
    with pytest.raises(ValueError, match='one-second'):
        task.one_second_features(np.zeros(48000, dtype=np.float32))


def test_recording_reconstruction_rejects_false_filename_continuity():
    prepare = script('prepare_external_v3')
    left = np.arange(12, dtype=np.float32)
    right = np.arange(6, 18, dtype=np.float32)
    assert prepare.matching_tail(left, right, 6)
    right[0] += .001
    assert not prepare.matching_tail(left, right, 6)
    assert not prepare.matching_tail(left[:2], right, 6)


def test_external_panel_rejects_post_freeze_threshold_changes(prepared):
    runner = script('evaluate_external_v3')
    model = prepared['model']
    arm = {'name': 'software-control', 'bundle': str(prepared['bundle']),
           'model_json_sha256': file_hash(prepared['bundle'] / 'model.json'),
           'model_id': model.model_id, 'decision': copy.deepcopy(model.metadata['decision'])}
    assert runner.verify_panel({'models': [arm]})['software-control'].model_id == model.model_id
    arm['decision']['on_hits'] += 1
    with pytest.raises(ValueError, match='decision changed'):
        runner.verify_panel({'models': [arm]})
