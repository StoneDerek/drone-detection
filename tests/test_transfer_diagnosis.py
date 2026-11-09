"""Software controls for numerical normalization and structural probes."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from drone_detector.audio import normalize_context_gain
from drone_detector.spectrogram import SpectrogramConfig


def load_script(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_rms_is_a_true_numerical_reference_without_hidden_peak_clipping():
    wave = np.zeros(16000, dtype=np.float32)
    wave[:8] = .01
    exact = normalize_context_gain(wave, SpectrogramConfig(normalize_rms='exact'))
    bounded = normalize_context_gain(wave, SpectrogramConfig(normalize_rms=True))
    assert np.sqrt(np.mean(exact.astype(np.float64) ** 2)) == pytest.approx(.05, abs=1e-8)
    assert np.max(exact) > 1  # Numerical feature control, not clipped PCM playback.
    assert np.sqrt(np.mean(bounded.astype(np.float64) ** 2)) < .05


def test_exact_rms_keeps_silence_finite_and_rejects_unknown_modes():
    zero = np.zeros(16000, dtype=np.float32)
    np.testing.assert_array_equal(normalize_context_gain(zero, SpectrogramConfig(normalize_rms='exact')), zero)
    with pytest.raises(ValueError, match='amplitude'):
        SpectrogramConfig(normalize_rms='unrecognized')


def test_boolean_gain_policy_keeps_its_previous_formula():
    wave = np.linspace(-.03, .03, 16000, dtype=np.float32)
    cfg = SpectrogramConfig(normalize_rms=True)
    rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    expected = (wave * min(.05 / rms, 10., .98 / float(np.max(np.abs(wave))))).astype(np.float32)
    np.testing.assert_array_equal(normalize_context_gain(wave, cfg), expected)


def test_time_permutations_keep_band_marginals_but_can_break_synchrony():
    script = load_script('transfer_diagnosis_v1')
    image = np.tile(np.arange(20, dtype=np.float32), (5, 1))
    probes = script.tensor_probes(image)
    for key in ['common_time_permutation', 'independent_band_time_permutation']:
        for value in probes[key]:
            np.testing.assert_array_equal(np.sort(value, axis=1), image)
    assert np.array_equal(probes['common_time_permutation'][0][0], probes['common_time_permutation'][0][1])
    assert not np.array_equal(probes['independent_band_time_permutation'][0][0], probes['independent_band_time_permutation'][0][1])


def test_training_weights_balance_intervals_and_classes_without_validation_data():
    script = load_script('learnability_normalization_v1')
    items = [{'file': f'{index}', 'group': group, 'label': label} for index, (group, label) in enumerate(
        [('p1', 'drone'), ('p2', 'drone'), ('p2', 'drone'), ('p2', 'drone'),
         ('n1', 'background'), ('n1', 'background'), ('n1', 'background'), ('held', 'background')])]
    assignment = {item['file']: 'train' if item['group'] != 'held' else 'val' for item in items}
    ids, weights = script.interval_class_weights(items, assignment, 'train')
    assert ids.tolist() == list(range(7))
    assert weights[7] == 0
    assert weights[:4].sum() == pytest.approx(weights[4:7].sum())
    assert weights[0] == pytest.approx(weights[1:4].sum())
