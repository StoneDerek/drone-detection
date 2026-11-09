import numpy as np
import pytest
from dataclasses import replace

from drone_detector.common import file_hash, write_json
from drone_detector.model import load_model, save_model
from drone_detector.spectral_training import train_cnn
from drone_detector.spectral_calibration import freeze_calibration_metadata
from drone_detector.spectrogram import SpectrogramConfig, axes, spectrogram
from drone_detector.stream import replay_audio


def tone(seconds=3):
    t = np.arange(round(seconds * 16000)) / 16000
    return (.2 * np.sin(2 * np.pi * 500 * t) + .002 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)


def test_frequency_axes_shapes_and_intensity_are_distinct():
    config = SpectrogramConfig()
    wave = tone()
    image = spectrogram(wave, config)
    freq, times, _ = axes(config)
    assert image.shape == (1933, 172) == (len(freq), len(times))
    assert np.allclose(np.diff(freq), 16000 / 4096)
    low = np.argmin(abs(freq - 500))
    harmonic = np.argmin(abs(freq - 1000))
    assert image[low].mean() > image[harmonic].mean() > .3
    power = spectrogram(wave, SpectrogramConfig(intensity='linear_power'))
    assert power[harmonic].mean() < .001
    mel = SpectrogramConfig(axis='mel')
    assert spectrogram(wave, mel).shape == (64, 172)
    assert not np.allclose(np.diff(axes(mel)[0]), np.diff(axes(mel)[0])[0])
    matched = SpectrogramConfig(axis='linear_filterbank', mel_bins=256)
    assert spectrogram(wave, matched).shape == (256, 172)
    assert np.allclose(np.diff(axes(matched)[0]), np.diff(axes(matched)[0])[0])
    assert spectrogram(wave * .01, config) == pytest.approx(image, abs=2e-3)
    assert np.count_nonzero(spectrogram(np.zeros(1000), config)) == 0
    with pytest.raises(ValueError):
        spectrogram(np.array([]), config)


def test_reference_rms_reaches_spectrogram_only_with_a_fixed_power_reference():
    config = SpectrogramConfig(axis='linear_filterbank', mel_bins=32)
    wave = tone() * .01
    relative = spectrogram(wave, config)
    normalized_relative = spectrogram(wave, replace(config, normalize_rms=True))
    np.testing.assert_allclose(relative, normalized_relative, atol=2e-3)
    fixed = replace(config, power_reference='unit_stft_power')
    raw_image = spectrogram(wave, fixed)
    normalized_image = spectrogram(wave, replace(fixed, normalize_rms=True))
    assert np.mean(abs(raw_image - normalized_image)) > .01
    # Shared reference makes a gain change observable before the floor/clamp.
    assert np.mean(abs(spectrogram(wave * 2, fixed) - raw_image)) > .001
    assert not np.any(spectrogram(np.zeros(16000), replace(fixed, normalize_rms=True)))
    for kwargs in [{'power_reference': 'unknown'}, {'normalize_rms': 'yes'}, {'gain_target_rms': 0}]:
        with pytest.raises(ValueError):
            replace(config, **kwargs)


@pytest.mark.parametrize('variant', ['compact', 'frequency_context', 'coverage', 'coverage_medium'])
def test_cnn_training_artifact_streaming_and_gradcam(tmp_path, variant):
    config = SpectrogramConfig(axis='mel', n_fft=1024)
    cache = tmp_path / 'cache'
    cache.mkdir()
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('software test only\n')
    records = []
    for split in ['train', 'development']:
        for label in [0, 1]:
            name = f'{split}-{label}'
            wave = tone() if label else np.random.default_rng(9).normal(0, .02, 48000).astype(np.float32)
            np.save(cache / f'{name}.npy', np.stack([spectrogram(wave, config)] * 2))
            records.append({'id': name, 'label': 'drone' if label else 'non_drone', 'split': split,
                            'cache': f'{name}.npy', 'origin': 'synthetic'})
    plan = {'manifest_sha256': file_hash(manifest), 'code': {'software_test': True}}
    write_json(cache / 'index.json', {'identity': {'manifest_sha256': file_hash(manifest)},
                                     'shape': [64, 184], 'records': records})
    model = train_cnn(manifest, cache, config, tmp_path / 'raw', plan, epochs=2, variant=variant)
    assert model.metadata['training']['origins'] == ['synthetic']
    assert model.metadata['architecture']['parameters'] == {'compact': 1781, 'frequency_context': 3077,
        'coverage': 1781, 'coverage_medium': 93313}[variant]
    metadata = freeze_calibration_metadata(model.metadata,
        {'on_threshold': .5, 'off_threshold': .375, 'on_hits': 2, 'off_hits': 2},
        {'scope': 'software test only'}, {'split': 'development'})
    model = save_model(tmp_path / 'frozen', metadata, model.arrays)
    assert load_model(tmp_path / 'frozen', require_frozen=True).metadata['status'] == 'frozen'
    explanation = model.gradcam(tone(), layer='conv1')
    assert explanation['heatmap'].shape == explanation['spectrogram'].shape == (64, 184)
    assert np.isfinite(explanation['heatmap']).all()
    assert explanation['classifier_score'] == pytest.approx(model.score(tone()), abs=1e-6)
    first = replay_audio(model, np.tile(tone(), 2))
    second = replay_audio(model, np.tile(tone(), 2), block_samples=1379)
    assert [r['score'] for r in first] == pytest.approx([r['score'] for r in second])
    batched = model.batch_trace(np.tile(tone(), 2))
    assert [r['score'] for r in first] == pytest.approx([r['score'] for r in batched], abs=1e-6)
    assert [r['alarm'] for r in first] == [r['alarm'] for r in batched]
    assert [r['timestamp_seconds'] for r in first] == [r['timestamp_seconds'] for r in batched]
    assert all(r['compute_seconds'] is None for r in batched)
    path = tmp_path / 'frozen' / 'parameters.npz'
    path.write_bytes(path.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_model(path.parent, True)
