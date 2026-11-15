"""Asset/input contract tests; actual-checkpoint evidence is a separate smoke run."""

import numpy as np
import pytest

from drone_detector import clap_encoder
from drone_detector.clap_encoder import CLAPEncoder, verify_assets
from drone_detector.common import write_json


@pytest.mark.parametrize('audio', [np.zeros((2, 8)), np.array([]), np.zeros(160001),
                                   np.array([np.nan]), np.array([np.inf])])
def test_bad_audio_rejected_before_model_loading(audio):
    encoder = object.__new__(CLAPEncoder)
    with pytest.raises(ValueError):
        encoder.features(audio)


def test_asset_manifest_must_match_pinned_source(tmp_path):
    write_json(tmp_path / 'assets.json', {'schema_version': 1, 'model': 'other', 'revision': 'mutable'})
    with pytest.raises(ValueError, match='Unknown CLAP asset manifest'):
        verify_assets(tmp_path)


def test_changed_asset_rejected(tmp_path, monkeypatch):
    import hashlib
    expected = b'original'
    monkeypatch.setattr(clap_encoder, 'ASSETS', {
        'config.json': (len(expected), hashlib.sha256(expected).hexdigest())})
    write_json(tmp_path / 'assets.json', {
        'schema_version': 1, 'model': clap_encoder.MODEL_NAME,
        'revision': clap_encoder.MODEL_REVISION,
        'sha256': {'config.json': hashlib.sha256(expected).hexdigest()},
    })
    (tmp_path / 'config.json').write_bytes(b'changed!')
    with pytest.raises(ValueError, match='missing or changed'):
        verify_assets(tmp_path)


def test_long_audio_is_rejected_instead_of_randomly_cropped():
    encoder = object.__new__(CLAPEncoder)
    encoder._load = lambda: pytest.fail('Long audio must not reach random native truncation')
    with pytest.raises(ValueError, match='no longer than 10 seconds'):
        encoder.features(np.zeros(160001, dtype=np.float32))


def test_actual_clap_native_projection_is_repeatable():
    import os
    assets = os.environ.get('CLAP_TEST_ASSETS')
    if not assets:
        pytest.skip('Set CLAP_TEST_ASSETS to verify the actual local checkpoint')
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from scipy.signal import resample_poly
    torch.set_num_threads(2)
    encoder = CLAPEncoder(assets)
    audio = np.random.default_rng(42).normal(0, .01, 48000).astype(np.float32)
    before = torch.get_rng_state().clone()
    first = encoder.features(audio)
    assert torch.equal(before, torch.get_rng_state())
    np.testing.assert_array_equal(first, encoder.features(audio))
    assert not encoder._model.training
    assert not any(p.requires_grad for p in encoder._model.parameters())
    native = encoder._extractor(resample_poly(audio, 3, 1), sampling_rate=48000,
                                truncation='rand_trunc', padding='repeatpad',
                                max_length=480000, return_tensors='pt')
    with torch.inference_mode():
        expected = torch.nn.functional.normalize(encoder._model(**native).audio_embeds, dim=-1)
    np.testing.assert_array_equal(first, expected[0].numpy().astype(np.float64))
    assert first.shape == (512,) and np.isfinite(first).all()
    assert np.isclose(np.linalg.norm(first), 1, atol=1e-6)
