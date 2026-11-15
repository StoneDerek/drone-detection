"""Integrity and native-frontend checks; actual weights are opt-in, never fetched by tests."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from drone_detector import beats_encoder as beats
from drone_detector.common import write_json


def test_asset_identity_cannot_be_overridden_by_manifest(tmp_path, monkeypatch):
    code = b"pinned source\n"
    model = b"pinned weights"
    sha = lambda value: hashlib.sha256(value).hexdigest()  # noqa: E731
    monkeypatch.setattr(beats, "SOURCE_ASSETS", {"BEATs.py": (len(code), sha(code))})
    monkeypatch.setattr(beats, "CHECKPOINT_BYTES", len(model))
    monkeypatch.setattr(beats, "CHECKPOINT_SHA256", sha(model))
    (tmp_path / "BEATs.py").write_bytes(code)
    (tmp_path / "model.pt").write_bytes(model)
    manifest = {
        "schema_version": 1, "model": beats.MODEL_NAME,
        "sha256": {"BEATs.py": sha(code), "model.pt": sha(model)},
    }
    write_json(tmp_path / "assets.json", manifest)
    beats.verify_assets(tmp_path)
    (tmp_path / "model.pt").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="asset missing or changed"):
        beats.verify_assets(tmp_path)
    manifest["sha256"]["model.pt"] = sha(b"changed weights")
    write_json(tmp_path / "assets.json", manifest)
    with pytest.raises(ValueError, match="pinned identities"):
        beats.verify_assets(tmp_path)


@pytest.mark.parametrize("audio", [np.zeros(2799), np.zeros((1, 48000)), np.array([])])
def test_short_and_multichannel_inputs_need_explicit_preparation(audio):
    encoder = object.__new__(beats.BEATsEncoder)
    with pytest.raises(ValueError, match="mono 16 kHz"):
        encoder.features(audio)


def test_nonfinite_audio_is_rejected_before_model_loading():
    encoder = object.__new__(beats.BEATsEncoder)
    audio = np.zeros(48000)
    audio[100] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        encoder.features(audio)


def test_real_encoder_preserves_native_frontend_and_frozen_features():
    directory = os.environ.get("BEATS_TEST_ASSETS")
    if not directory:
        pytest.skip("Set BEATS_TEST_ASSETS to opt in to actual local checkpoint verification")
    torch = pytest.importorskip("torch")
    kaldi = pytest.importorskip("torchaudio.compliance.kaldi")
    torch.set_num_threads(2)
    encoder = beats.BEATsEncoder(Path(directory))
    audio = np.random.default_rng(42).normal(0, 0.01, 48000).astype(np.float32)
    before = torch.get_rng_state().clone()
    result = encoder.features(audio)
    assert torch.equal(before, torch.get_rng_state())
    assert result.shape == (768,) and np.isfinite(result).all()
    assert not encoder._model.training
    assert not any(parameter.requires_grad for parameter in encoder._model.parameters())
    assert encoder._model.cfg.finetuned_model is False
    tensor = torch.from_numpy(audio)[None]
    reference_fbank = kaldi.fbank(
        tensor * 32768, num_mel_bins=128, sample_frequency=16000,
        frame_length=25, frame_shift=10,
    )
    reference_fbank = (reference_fbank - 15.41663) / (2 * 6.55582)
    with torch.inference_mode():
        torch.testing.assert_close(encoder._model.preprocess(tensor)[0], reference_fbank, rtol=0, atol=0)
        native_tokens = encoder._model.extract_features(tensor)[0]
    expected = native_tokens.mean(dim=1)[0].numpy().astype(np.float64)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result, encoder.features(audio))
    # Exact native minimum must work without invisible waveform padding.
    assert encoder.features(np.zeros(encoder.minimum_samples)).shape == (768,)
