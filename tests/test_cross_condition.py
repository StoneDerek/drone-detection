import importlib.util
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from drone_detector.common import file_hash, write_json


def module(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_sampling_uses_only_complete_unique_contexts_and_spans_the_recording():
    evaluate = module('cross_condition_v1')
    assert evaluate.context_starts(15999).tolist() == []
    assert evaluate.context_starts(16000).tolist() == [0]
    starts = evaluate.context_starts(16000 * 100)
    assert len(starts) == 60 and len(np.unique(starts)) == 60
    assert starts[0] == 0 and starts[-1] + 16000 == 16000 * 100
    assert np.all(starts % 8000 == 0)


def test_primary_label_parser_does_not_promote_a_secondary_drone_comment(tmp_path):
    fetch = module('fetch_kenya_positive_v1')
    path = tmp_path / 'source.xlsx'
    prefix = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    strings = ['Filename', 'Duration', 'Label', 'other.wav', 'Axe', 'Drone', 'primary.wav']
    shared = f'<sst xmlns="{prefix}">' + ''.join(f'<si><t>{value}</t></si>' for value in strings) + '</sst>'
    rows = []
    for number, values in enumerate([[0, 1, 2], [3, 1, 4, 5], [6, 1, 5]], 1):
        cells = ''.join(f'<c r="{chr(65+i)}{number}" t="s"><v>{index}</v></c>' for i, index in enumerate(values))
        rows.append(f'<row r="{number}">{cells}</row>')
    with ZipFile(path, 'w') as archive:
        archive.writestr('xl/sharedStrings.xml', shared)
        archive.writestr('xl/worksheets/sheet2.xml', f'<worksheet xmlns="{prefix}"><sheetData>{"".join(rows)}</sheetData></worksheet>')
    assert fetch.primary_drone_labels(path) == ['primary.wav']


def test_fixed_non_neural_package_rejects_late_parameter_changes(tmp_path):
    evaluate = module('cross_condition_v1')
    write_json(tmp_path / 'model.json', {'family': 'software-control'})
    params = tmp_path / 'parameters.npz'
    np.savez_compressed(params, mean=np.zeros(512), scale=np.ones(512), coef=np.zeros(512), intercept=np.zeros(1))
    arm = {'name': 'control', 'family': 'logistic', 'bundle': str(tmp_path), 'threshold': .5,
           'metadata_sha256': file_hash(tmp_path / 'model.json'), 'parameters_sha256': file_hash(params)}
    assert 'control' in evaluate.fixed_models({'models': [arm]})
    np.savez_compressed(params, mean=np.ones(512))
    with pytest.raises(ValueError, match='changed'):
        evaluate.fixed_models({'models': [arm]})


def test_positive_only_cohort_has_no_negative_rate_or_auc():
    evaluate = module('cross_condition_v1')
    rows = [{'id': str(i), 'source': 'new', 'group_id': 'same-original', 'label': 'drone',
             'sampled_contexts': 9, 'silent_contexts': 0, 'predicted': 1, 'max_score': .9} for i in range(5)]
    result = evaluate.source_metrics(rows)['new']
    assert result['conservative_groups'] == 1
    assert result['negative_files'] == 0 and result['max_score_auc'] is None
    assert result['positive_files_detected'] == 5


def test_partial_source_fetch_checks_original_hash_and_retains_bit_exact_pcm(tmp_path, monkeypatch):
    import soundfile as sf

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    fetch = module('controlled_source_v1')
    monkeypatch.setattr(fetch, 'ROOT', tmp_path / 'raw')
    monkeypatch.setattr(fetch, 'REPORT', tmp_path / 'report')
    fetch.ROOT.mkdir()
    fetch.REPORT.mkdir()
    pcm = (np.arange(48000) % 173 - 86).astype('int16')
    stream = io.BytesIO()
    sf.write(stream, pcm, 48000, format='WAV', subtype='PCM_16')
    raw = stream.getvalue()
    metadata = json.dumps({'identification': {'recording_id': 'test'}, 'target': {'present': False}}).encode()
    content = {'original.wav': raw, 'original.json': metadata}

    class Archive:
        def read(self, path):
            return content[path]

    monkeypatch.setattr(fetch, 'remote_zip', lambda plan: Archive())
    item = {'publisher': {'recording_id': 'test', 'has_uav': 'false', 'data_origin': 'self_collected',
                          'scene_type': 'test', 'acquisition_batch_id': 'batch-undated', 'session_id': 'placeholder'},
            'sample_count': 48000,
            'wav': {'path': 'original.wav', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()},
            'json': {'path': 'original.json', 'bytes': len(metadata), 'sha256': hashlib.sha256(metadata).hexdigest()}}
    row = fetch.fetch_one(item, {})
    restored, rate = sf.read(row['path'], dtype='int16')
    assert rate == 48000 and np.array_equal(restored, pcm)
    assert row['source'] == 'nanjing-field' and row['label'] == 'non_drone'
    assert row['group_id'] == 'nanjing-batch-undated'
    assert row['original_wav_sha256'] == item['wav']['sha256']
    assert fetch.fetch_one(item, {}) == row
    item['wav']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='changed'):
        fetch.fetch_one(item, {})


def test_partial_source_fetch_rejects_wrong_publisher_hash_before_audio_storage(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    fetch = module('controlled_source_v1')
    monkeypatch.setattr(fetch, 'ROOT', tmp_path)
    class Archive:
        def read(self, path):
            return b'changed'
    monkeypatch.setattr(fetch, 'remote_zip', lambda plan: Archive())
    item = {'publisher': {'recording_id': 'test'},
            'json': {'path': 'test.json', 'bytes': 7, 'sha256': '0' * 64},
            'wav': {'path': 'test.wav', 'bytes': 7, 'sha256': '0' * 64}}
    with pytest.raises(ValueError, match='SHA256'):
        fetch.fetch_one(item, {})
    assert not (tmp_path / 'audio').exists()


def test_range_retry_honors_server_delay_and_restores_read_offset(monkeypatch):
    from email.message import Message
    from urllib.error import HTTPError

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    fetch = module('controlled_source_v1')
    reader = fetch.PatientRangeReader('https://example.invalid/test', 100)
    reader.seek(5)
    positions, delays = [], []
    headers = Message()
    headers['Retry-After'] = '90'

    def read(self, size):
        positions.append(self.position)
        if len(positions) == 1:
            self.position = 20
            raise HTTPError(self.url, 429, 'rate limit', headers, None)
        self.position += size
        return b'abc'

    monkeypatch.setattr(fetch.RangeReader, 'read', read)
    monkeypatch.setattr(fetch.time, 'sleep', delays.append)
    assert reader.read(3) == b'abc'
    assert positions == [5, 5] and reader.position == 8
    assert delays == [30, 30, 30]


def test_unknown_capture_dates_are_pooled_across_classes_before_scoring(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    refine = module('refine_controlled_groups_v1')
    monkeypatch.setattr(refine, 'ROOT', tmp_path)
    (tmp_path / 'source-records').mkdir()
    (tmp_path / 'source-records/sessions.csv').write_text('pinned test source')
    write_json(tmp_path / 'plan.json', {'models': 'unchanged'})
    sessions = [{'session_id': 'unknown1', 'acquisition_date': ''},
                {'session_id': 'unknown2', 'acquisition_date': ''},
                {'session_id': 'known', 'acquisition_date': '2026-10-12'}]
    records = [{'recording_id': 'positive', 'session_id': 'unknown1'},
               {'recording_id': 'negative', 'session_id': 'unknown2'},
               {'recording_id': 'dated', 'session_id': 'known'}]
    monkeypatch.setattr(refine, 'csv_rows', lambda name: sessions if name == 'sessions' else records)
    monkeypatch.setattr('sys.argv', ['refine', 'freeze'])
    refine.main()
    result = json.loads((tmp_path / 'grouping-amendment.json').read_text())['assignment']
    assert result['positive'] == result['negative']
    assert result['positive'] != result['dated']
    write_json(tmp_path / 'first-use.json', {})
    with pytest.raises(ValueError, match='before first'):
        refine.main()
