import hashlib
import importlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    module = importlib.import_module('prepare_point_cache')
    sources, output, cache = tmp_path / 'sources', tmp_path / 'output', tmp_path / 'cache'
    sources.mkdir()
    rows, payloads = [], {}
    source = sources / 'archive.zip'
    with zipfile.ZipFile(source, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for number in [1, 2]:
            buffer = io.BytesIO()
            sf.write(buffer, np.array([.25, 1.5, -.5], dtype=np.float32), 44100,
                     format='WAV', subtype='FLOAT')
            payload = buffer.getvalue()
            name = f'{number:02d}-36-S-15m-Mic1.wav'
            payloads[name] = payload
            archive.writestr(name, payload)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            rows.append({'source_archive': 'archive.zip', 'source_archive_sha256': checksum,
                         'member_path': info.filename, 'member_bytes': info.file_size,
                         'member_crc32': info.CRC, 'member_sha256': None,
                         'sample_rate': 44100, 'native_frames': 3, 'duration_seconds': 3 / 44100,
                         'input_kind': 'point_microphone', 'split': None,
                         'accepted_for_training': False, 'accepted_for_blind_evaluation': False})
    index, lock = tmp_path / 'index.jsonl', tmp_path / 'lock.json'
    index.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    lock.write_text(json.dumps({'sources': [{'filename': 'archive.zip', 'bytes': source.stat().st_size, 'sha256': checksum}]}))
    return module, sources, output, cache, index, lock, payloads


def test_preparation_preserves_native_bytes_and_recovers_completed_files(tmp_path, monkeypatch):
    module, sources, output, cache, index, lock, originals = fixture(tmp_path, monkeypatch)
    summary = module.prepare(sources, output, cache, index, lock, .01)
    assert summary['completed_point_files'] == 2 and summary['native_bytes_unchanged']
    rows = [json.loads(line) for line in (output / 'recordings.jsonl').read_text().splitlines()]
    for row in rows:
        assert (output / row['path']).read_bytes() == originals[row['member_path']]
        assert row['quality']['peak'] == 1.5 and not row['accepted_for_training']
    # Stop between final file publication and completion receipt; the next
    # invocation must recover it without downloading the source again.
    path = output / rows[0]['path']
    complete = path.with_name(path.name + '.complete.json')
    complete.rename(path.with_name(path.name + '.pending.json'))
    def no_stage(*_):
        pytest.fail('Completed point members should resume without restaging originals')
    monkeypatch.setattr(module, 'stage_archive', no_stage)
    resumed = module.prepare(sources, output, cache, index, lock, .01)
    assert resumed == summary and complete.exists()
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='cache changed'):
        module.prepare(sources, output, cache, index, lock, .01)


def test_changed_archive_and_array_inputs_stop_before_copy(tmp_path, monkeypatch):
    module, sources, output, cache, index, lock, _ = fixture(tmp_path, monkeypatch)
    source = sources / 'archive.zip'
    original = source.read_bytes()
    source.write_bytes(b'X' + original[1:])
    with pytest.raises(ValueError, match='source lock/storage'):
        module.prepare(sources, output, cache, index, lock, .01)
    assert not list((output / 'native-point').glob('*.wav'))
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    rows[0]['member_path'] = rows[0]['member_path'].replace('Mic1', 'Arr1')
    index.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='individual point microphones'):
        module.prepare(sources, output, cache, index, lock, .01)
