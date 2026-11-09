import numpy as np
import pytest
from types import SimpleNamespace

from drone_detector.audio import FeatureConfig, features, normalize_context_gain, quality_measures
from drone_detector.stream import replay_audio


def sine(frequency, amplitude=.1, seconds=3):
    t = np.arange(round(16000 * seconds)) / 16000
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def test_quality_measurement_distinguishes_low_frequency_energy():
    assert 'low_frequency_dominance' in quality_measures(sine(30), 16000)['quality_flags']
    assert 'low_frequency_dominance' not in quality_measures(sine(250), 16000)['quality_flags']
    assert quality_measures(np.zeros(48000), 16000)['quality_flags'] == []
    with pytest.raises(ValueError):
        quality_measures(np.array([]), 16000)


def test_quality_detects_clipped_waveform():
    wave = np.clip(sine(200, amplitude=2), -1, 1)
    assert 'clipping' in quality_measures(wave, 16000)['quality_flags']


def test_context_gain_target_cap_and_silence_guard():
    config = FeatureConfig(representation='yamnet_rms')
    normal = normalize_context_gain(sine(200), config)
    assert np.sqrt(np.mean(normal**2)) == pytest.approx(.05, rel=1e-5)
    faint = sine(200, amplitude=.0001)
    np.testing.assert_allclose(normalize_context_gain(faint, config), faint * 10)
    silence = sine(200, amplitude=1e-7)
    np.testing.assert_array_equal(normalize_context_gain(silence, config), silence)
    spiky = np.zeros(48000, dtype=np.float32)
    spiky[0] = .99
    assert np.max(np.abs(normalize_context_gain(spiky, config))) <= .980001


def test_gain_uses_available_audio_before_padding():
    captured = []
    class Encoder:
        def features(self, wave):
            captured.append(wave)
            return np.zeros(1024)
    features(sine(200, seconds=1), FeatureConfig(representation='yamnet_rms'), Encoder())
    wave = captured[0]
    assert len(wave) == 48000 and np.count_nonzero(wave[16000:]) == 0
    assert np.sqrt(np.mean(wave[:16000]**2)) == pytest.approx(.05, rel=1e-5)


def test_quality_advisories_do_not_change_binary_alarm(prepared):
    model = prepared['model']
    audio = sine(30, seconds=6)
    rows = replay_audio(model, audio)
    assert all('low_frequency_dominance' in r['quality_flags'] for r in rows)
    assert all(r['score'] == pytest.approx(model.score(audio[round(r['window_start_seconds']*16000):round(r['timestamp_seconds']*16000)])) for r in rows)


def test_quality_flags_preserve_active_alarm():
    model = SimpleNamespace(config=FeatureConfig(), model_id='software-test', score=lambda samples: 1.,
                            metadata={'decision': {'on_threshold': .5, 'off_threshold': .375,
                                                   'on_hits': 2, 'off_hits': 2}})
    rows = replay_audio(model, sine(30, seconds=6), block_samples=1379)
    assert not rows[0]['alarm'] and all(row['alarm'] for row in rows[1:])
    assert all('low_frequency_dominance' in row['quality_flags'] for row in rows)
