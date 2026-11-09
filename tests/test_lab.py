import base64
import http.client
import io
import json
import threading

import numpy as np
import pytest
import soundfile as sf

from drone_detector.common import write_json
from drone_detector.lab import make_lab_server
from drone_detector.model import save_model
from drone_detector.spectrogram import SpectrogramConfig, build_cnn


@pytest.fixture
def lab_server(prepared, tmp_path):
    config = SpectrogramConfig(axis='mel', n_fft=1024)
    net = build_cnn()
    weights = net.get_weights()
    metadata = {'schema_version': 2, 'status': 'frozen', 'family': 'spectrogram_cnn',
        'features': config.to_dict(), 'class_mapping': {'0': 'non_drone', '1': 'drone'},
        'weight_shapes': [list(w.shape) for w in weights], 'training': {'origins': ['synthetic']},
        'limitations': 'Random-weight software fixture; no detection evidence',
        'decision': prepared['model'].metadata['decision'], 'experimental': True,
        'selection': {'chosen': {'metrics': {'confusion': {'tp': 0, 'fp': 0},
                                            'positive_recordings': 1, 'negative_recordings': 1}}}}
    save_model(tmp_path / 'cnn', metadata, {f'w{i}': w for i, w in enumerate(weights)})
    catalog = tmp_path / 'catalog.json'
    write_json(catalog, {'models': [{'name': 'cnn', 'bundle': 'cnn'},
                                   {'name': 'non-neural', 'bundle': str(prepared['bundle'])}],
        'defaults': ['cnn', 'non-neural'], 'scope': 'Software fixture only', 'data_summary': 'Synthetic controls'})
    server = make_lab_server(catalog, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(server, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=30)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    status, payload = response.status, response.read()
    connection.close()
    return status, payload


def audio():
    buffer = io.BytesIO()
    wave = .2 * np.sin(2 * np.pi * 500 * np.arange(64000) / 16000)
    sf.write(buffer, wave, 16000, format='WAV')
    return buffer.getvalue()


def test_lab_compares_models_and_computes_actual_gradcam(lab_server):
    code, payload = request(lab_server, 'GET', '/api/models')
    metadata = json.loads(payload)
    assert code == 200 and len(metadata['models']) == 2
    assert metadata['models'][0]['gradcam'] and not metadata['models'][1]['gradcam']
    code, payload = request(lab_server, 'POST', '/api/analyze?models=cnn,non-neural', audio())
    comparison = json.loads(payload)
    assert code == 200 and len(comparison['models']) == 2
    trace = comparison['models'][0]['trace']
    code, payload = request(lab_server, 'POST', '/api/explain?model=cnn&start=0&layer=conv1', audio())
    explanation = json.loads(payload)
    assert code == 200 and explanation['shape'] == [64, 184]
    assert len(base64.b64decode(explanation['heatmap_u8'])) == 64 * 184
    assert explanation['classifier_score'] == pytest.approx(trace[0]['score'], abs=1e-6)
    assert explanation['coarse_shape'] == [32, 92]
    assert len(base64.b64decode(explanation['native_heatmap_u8'])) == 32 * 92
    assert explanation['model_id'] == comparison['models'][0]['model_id']


def test_lab_rejects_inapplicable_attributions_and_invalid_audio(lab_server):
    assert request(lab_server, 'POST', '/api/explain?model=non-neural&start=0', audio())[0] == 422
    assert request(lab_server, 'POST', '/api/explain?model=cnn&start=nan', audio())[0] == 422
    assert request(lab_server, 'POST', '/api/analyze?models=cnn', b'broken')[0] == 422
    assert request(lab_server, 'POST', '/api/analyze?models=unknown', audio())[0] == 422
    assert request(lab_server, 'POST', '/api/analyze?models=cnn', audio(), {'Origin': 'https://untrusted.example'})[0] == 403
    silence = io.BytesIO()
    sf.write(silence, np.zeros(64000), 16000, format='WAV')
    code, payload = request(lab_server, 'POST', '/api/explain?model=cnn&start=0', silence.getvalue())
    assert code == 422 and 'silent input' in json.loads(payload)['error']
