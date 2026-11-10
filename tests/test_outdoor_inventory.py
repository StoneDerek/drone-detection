import hashlib
import importlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from drone_detector.source_inventory import outdoor_name, summarize_views, validate_completion_receipt

ROOT = Path(__file__).resolve().parents[1]


def views():
    return [{**outdoor_name(f'01-36-S-15m-{view}.wav'), 'complete_frames': 441000,
             'sample_rate': 44100, 'duration_seconds': 10.} for view in ['Mic1', 'Arr1', 'Arr2', 'Arr3']]


def test_simultaneous_array_means_do_not_inflate_point_exposure():
    summary = summarize_views(views())
    assert summary['wav_files'] == 4 and summary['measurement_groups'] == 1
    assert summary['point_microphone_files'] == 1 and summary['array_mean_files'] == 3
    assert summary['point_microphone_file_seconds'] == 10
    assert summary['all_view_file_seconds'] == 40
    assert summary['measurements'][0]['publisher_condition_code'] == 'S-15m'


def test_mirrors_missing_views_and_different_capture_extents_are_rejected():
    rows = views()
    with pytest.raises(ValueError, match='Repeated measurement/view'):
        summarize_views(rows + rows)
    with pytest.raises(ValueError, match='Missing recorder views'):
        summarize_views(rows[1:])
    rows[-1]['complete_frames'] -= 100
    with pytest.raises(ValueError, match='durations/rates differ'):
        summarize_views(rows)
    with pytest.raises(ValueError, match='Unrecognized'):
        outdoor_name('looks-positive.wav')


def test_unknown_first_acquisition_hash_still_requires_valid_plan_identity():
    item = {'filename': 'archive.zip', 'bytes': 10, 'sha256': None, 'url': 'https://example.invalid/a'}
    receipt = {'item_identity': hashlib.sha256(json.dumps(item, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
               'filename': item['filename'], 'bytes': item['bytes'], 'url': item['url'], 'sha256': 'a' * 64}
    validate_completion_receipt(item, receipt)
    with pytest.raises(ValueError, match='pinned intake plan'):
        validate_completion_receipt(item, {**receipt, 'bytes': 11})
    with pytest.raises(ValueError, match='pinned intake plan'):
        validate_completion_receipt(item, {**receipt, 'sha256': 'z' * 64})


def test_native_float_qa_preserves_values_outside_numeric_one(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    module = importlib.import_module('audit_native_recording_pairs')
    buffer = io.BytesIO()
    original = np.array([.25, 1.5, -.5], dtype=np.float32)
    sf.write(buffer, original, 44100, format='WAV', subtype='FLOAT')
    facts, samples = module.native_facts(buffer.getvalue())
    np.testing.assert_array_equal(samples[:, 0], original)
    assert facts['peak'] == 1.5 and facts['fraction_abs_above_one'] == 1 / 3
    assert facts['subtype'] == 'FLOAT' and facts['sample_rate'] == 44100
    bad = io.BytesIO()
    sf.write(bad, [0., np.nan], 44100, format='WAV', subtype='FLOAT')
    with pytest.raises(ValueError, match='nonfinite'):
        module.native_facts(bad.getvalue())
