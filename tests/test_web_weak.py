import http.client
import io
import json
import threading

import numpy as np
import pytest
import soundfile as sf

from drone_detector.audio import FeatureConfig, features
from drone_detector.common import file_hash, write_json
from drone_detector.stream import infer_file
from drone_detector.web import make_server
from drone_detector.weak import evaluate_weak, train_weak, tune_weak


@pytest.fixture
def web_server(prepared):
    server = make_server(prepared["bundle"], port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=15)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = response.read()
    code = response.status
    connection.close()
    return code, payload


def test_product_model_and_frontend(web_server):
    code, page = request(web_server, "GET", "/")
    assert code == 200 and b"Inspect a recording" in page
    code, raw = request(web_server, "GET", "/api/model")
    model = json.loads(raw)
    assert code == 200 and "Synthetic software-test" in model["scope"]
    assert model["features"]["window_seconds"] == 3
    assert "do not estimate wind speed" in model["quality_advisories"]
    assert b"Quality flags advise review and do not suppress an alarm" in page


def test_upload_matches_offline_scores(web_server, prepared, tmp_path):
    row = next(r for r in prepared["rows"] if r["split"] == "development")
    path = prepared["manifest"].parent / row["path"]
    code, raw = request(web_server, "POST", "/api/analyze", path.read_bytes())
    result = json.loads(raw)
    assert code == 200 and result["model_id"] == prepared["model"].model_id
    expected, _ = infer_file(prepared["model"], path)
    assert len(result["trace"]) == len(expected) > 0
    for a, b in zip(result["trace"], expected, strict=True):
        for field in ["score", "alarm", "timestamp_seconds", "reason", "padded",
                      "quality_flags", "low_frequency_fraction_below_100hz", "window_clipped_fraction"]:
            assert a[field] == b[field]


def test_invalid_and_excess_uploads_are_actionable(web_server):
    code, raw = request(web_server, "POST", "/api/analyze", b"not an audio recording")
    assert code == 422 and "Unable to decode" in json.loads(raw)["error"]
    code, _ = request(web_server, "POST", "/api/analyze", b"")
    assert code == 413
    code, _ = request(web_server, "POST", "/api/analyze", b"", {"Content-Length": str(65 * 1024 * 1024)})
    assert code == 413
    code, _ = request(web_server, "POST", "/api/analyze", b"test", {"Origin": "https://untrusted.example"})
    assert code == 403


def test_product_silence_and_short_input(web_server):
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(1600, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
    code, raw = request(web_server, "POST", "/api/analyze", buffer.getvalue())
    result = json.loads(raw)
    assert code == 200 and result["trace"][0]["score"] is None
    assert result["trace"][0]["reason"] == "insufficient_audio" and not result["alarms"]
    assert any("Short recording" in w for w in result["warnings"])


@pytest.mark.parametrize("frequency,amplitude,flag", [
    (30, .1, "low_frequency_dominance"), (250, 2, "clipping")])
def test_quality_advisories_reach_upload_response(web_server, frequency, amplitude, flag):
    samples = np.clip(amplitude * np.sin(2 * np.pi * frequency * np.arange(64000) / 16000), -1, 1)
    buffer = io.BytesIO()
    sf.write(buffer, samples, 16000, format="WAV", subtype="PCM_16")
    code, raw = request(web_server, "POST", "/api/analyze", buffer.getvalue())
    result = json.loads(raw)
    assert code == 200
    assert all(flag in row["quality_flags"] for row in result["trace"])
    assert result["warnings"]
    if flag == "low_frequency_dominance":
        assert any("not a wind-speed" in warning for warning in result["warnings"])


def test_silent_recording_is_explicit(web_server):
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(64000), 16000, format="WAV", subtype="PCM_16")
    code, raw = request(web_server, "POST", "/api/analyze", buffer.getvalue())
    result = json.loads(raw)
    assert code == 200 and result["warnings"] == ["Silent input."]
    assert all(row["reason"] == "silence" and row["score"] == 0 for row in result["trace"])
    assert not result["alarms"]


def test_examples_cannot_escape_selection(prepared, tmp_path):
    selection = tmp_path / "selection.json"
    row = prepared["rows"][0]
    write_json(selection, {"recording_ids": [row["id"]], "manifest_sha256": file_hash(prepared["manifest"])})
    server = make_server(prepared["bundle"], port=0, manifest=prepared["manifest"], selection=selection)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert request(server, "GET", "/api/example/0")[0] == 200
        assert request(server, "GET", "/api/example/-1")[0] == 404
        assert request(server, "GET", "/api/example/../../LICENSE")[0] == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_pretrained_path_uses_explicit_encoder_and_same_context():
    class Encoder:
        def features(self, audio):
            assert audio.shape == (48000,) and audio.dtype == np.float32
            return np.ones(1024)

    config = FeatureConfig(representation="yamnet")
    result = features(np.ones(16000, dtype=np.float32) * .1, config, Encoder())
    assert result.shape == (1024,)
    with pytest.raises(ValueError, match="Missing frozen"):
        features(np.ones(48000), config)


def test_weak_train_freeze_and_recording_assessment(prepared, tmp_path):
    raw, frozen = tmp_path / "raw", tmp_path / "frozen"
    model = train_weak(prepared["manifest"], raw)
    assert "MI logistic" in model.metadata["training"]["method"]
    assert {r["id"] for r in model.metadata["training"]["records"]} == {
        r["id"] for r in prepared["rows"] if r["split"] == "train"}
    model = tune_weak(raw, prepared["manifest"], frozen)
    assert model.metadata["status"] == "frozen"
    sensitive = tune_weak(raw, prepared['manifest'], tmp_path / 'sensitive', policy='sensitivity-v2')
    assert 'lowest threshold' in sensitive.metadata['contract']['selection']
    assert sensitive.metadata['decision']['on_threshold'] <= model.metadata['decision']['on_threshold']
    report = evaluate_weak([frozen], prepared["manifest"], "final", tmp_path / "assessment")
    summary = report["results"][0]["summary"]
    assert summary["recordings"] > 0 and summary["positive_recordings"] > 0
    assert "Not assessed" in summary["event_metrics"]
    assert not summary["field_reliability_claim"]
    with pytest.raises(ValueError, match="already assessed"):
        evaluate_weak([frozen], prepared["manifest"], "final", tmp_path / "second")
