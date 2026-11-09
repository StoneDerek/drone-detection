import numpy as np
import pytest
import soundfile as sf

from drone_detector.audio import FeatureConfig, decode, features
from drone_detector.model import Model
from drone_detector.stream import Alarm, StreamDetector, alarm_intervals, infer_file, replay_audio


def tone(rate=16000, seconds=6.23):
    t = np.arange(round(rate * seconds)) / rate
    return (.15 * np.sin(2 * np.pi * 180 * t) + .03 * np.sin(2 * np.pi * 720 * t)).astype(np.float32)


def test_pcm_stereo_downmix_resample(tmp_path):
    wave = tone(22050, 1)
    path = tmp_path / "stereo.wav"
    sf.write(path, np.stack([wave, wave * .5], axis=1), 22050, subtype="PCM_16")
    audio, info = decode(path)
    assert info["channels"] == 2 and info["original_sample_rate"] == 22050
    assert len(audio) == 16000 and info["duration_seconds"] == 1
    assert np.max(np.abs(audio)) < .15
    assert np.sqrt(np.mean(audio**2)) == pytest.approx(np.sqrt(np.mean(wave**2)) * .75, rel=.01)


def test_nonfinite_file_rejected(tmp_path):
    path = tmp_path / "bad.wav"
    sf.write(path, np.array([0, np.nan, 1]), 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="nonfinite"):
        decode(path)


def test_relative_gain_invariance_and_absolute_shift():
    audio = tone(seconds=3)
    relative = FeatureConfig(representation="relative_mel")
    absolute = FeatureConfig()
    # Float32 FFT roundoff is amplified in near-null bands of a pure tone.
    # Require <0.005 dB discrepancy, rather than exact arithmetic invariance.
    np.testing.assert_allclose(features(audio, relative), features(audio * .1, relative), atol=.005)
    np.testing.assert_allclose(features(audio * .1, absolute)[:48] - features(audio, absolute)[:48],
                               -20, atol=.005)


@pytest.mark.parametrize("options", [{"hop_seconds": 0}, {"hop_seconds": 1e-10},
                                      {"f_max": 9000}, {"representation": "unknown"}])
def test_invalid_feature_configuration(options):
    with pytest.raises(ValueError):
        FeatureConfig(**options)


def test_file_and_block_stream_share_scores_and_states(prepared, tmp_path):
    model, audio = prepared["model"], tone()
    path = tmp_path / "tone.wav"
    sf.write(path, audio, 16000, subtype="FLOAT")
    offline, _ = infer_file(model, path)
    engine = StreamDetector(model)
    irregular = []
    start = 0
    for size in [1, 37, 160, 8000, 13, 32000, 101, 17000, 5000, 900000]:
        block = audio[start:start + size]
        irregular.extend(engine.push(block))
        start += len(block)
    irregular.extend(engine.flush())
    assert len(offline) == len(irregular) > 1
    for left, right in zip(offline, irregular, strict=True):
        for field in ["timestamp_seconds", "window_start_seconds", "padded", "alarm", "reason"]:
            assert left[field] == right[field]
        assert left["score"] == pytest.approx(right["score"], abs=1e-12)
        # Different delivery blocks can change physical availability, not the
        # feature windows or causal state. These times must not be conflated.
        assert right["delivered_at_seconds"] >= right["timestamp_seconds"]


def test_no_future_normalization(prepared):
    model = prepared["model"]
    audio = tone(seconds=7)
    changed = audio.copy()
    changed[5 * 16000:] *= .001
    first, second = replay_audio(model, audio), replay_audio(model, changed)
    for left, right in zip(first, second, strict=True):
        if left["timestamp_seconds"] <= 5:
            assert left["score"] == right["score"]


def test_silence_and_insufficient_input_are_explicit(prepared):
    model = prepared["model"]
    silence = replay_audio(model, np.zeros(16000, dtype=np.float32))
    assert len(silence) == 1 and silence[0]["score"] == 0 and not silence[0]["alarm"]
    assert silence[0]["padded"] and silence[0]["reason"] == "silence"
    short = replay_audio(model, tone(seconds=.1))
    assert short[0]["score"] is None and short[0]["reason"] == "insufficient_audio"
    assert replay_audio(model, np.empty(0)) == []


def test_flush_exact_end_and_new_tail(prepared):
    model = prepared["model"]
    exact = replay_audio(model, tone(seconds=6))
    tail = replay_audio(model, tone(seconds=6.1))
    assert len(exact) == 7 and not any(row["padded"] for row in exact)
    assert len(tail) == 8 and tail[-1]["padded"]
    assert tail[-1]["timestamp_seconds"] == 6.1
    engine = StreamDetector(model)
    engine.flush()
    with pytest.raises(ValueError, match="after flush"):
        engine.push(tone())
    with pytest.raises(ValueError, match="Already flushed"):
        engine.flush()


def test_bad_stream_input_rejected(prepared):
    engine = StreamDetector(prepared["model"])
    for samples in [np.zeros((2, 2)), np.array([np.inf])]:
        with pytest.raises(ValueError, match="finite"):
            engine.push(samples)
    with pytest.raises(ValueError, match="live-rate"):
        StreamDetector(prepared["model"], sample_rate=48000)


def test_label_mapping_positive_coefficient(prepared):
    base = prepared["model"]
    arrays = {k: value.copy() for k, value in base.arrays.items()}
    arrays["coef"][:] = 0
    arrays["intercept"][:] = 9
    model = Model(base.metadata, arrays)
    assert model.score(tone(seconds=3)) > .999


def test_hysteresis_and_merging():
    decision = {"on_threshold": .8, "off_threshold": .4, "on_hits": 2, "off_hits": 2}
    alarm = Alarm(decision)
    scores = [.9, .9, .5, .2, .2, .9, .9]
    states = [alarm.step(score) for score in scores]
    assert states == [False, True, True, True, False, False, True]
    rows = [{"alarm": state, "timestamp_seconds": 3 + .5 * i} for i, state in enumerate(states)]
    assert alarm_intervals(rows, 7, merge_gap=1) == [[3.5, 7]]
