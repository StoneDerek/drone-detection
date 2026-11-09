import hashlib
import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = load('fetch_source_intake')
catalog = load('audit_audrok_catalog')


class Response(io.BytesIO):
    def __init__(self, payload, status=200, headers=None):
        super().__init__(payload)
        self.status, self.headers = status, headers or {}


def item(data=b'original', checksum=True):
    return {'filename': 'source.zip', 'url': 'https://example.invalid/source',
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest() if checksum else None}


def test_first_hash_is_frozen_and_replacement_reuses_verified_source(tmp_path):
    source = item(checksum=False)
    with patch.object(fetch.urllib.request, 'urlopen', return_value=Response(b'original')) as request:
        result = fetch.acquire(source, tmp_path)
        assert not result['reused'] and 'first acquisition' in result['hash_status']
        # Simulate a stop after final rename, before completion publication.
        receipt = tmp_path / 'source.zip.complete.json'
        receipt.rename(tmp_path / 'source.zip.pending.json')
        resumed = fetch.acquire(source, tmp_path)
        assert resumed['reused'] and request.call_count == 1
    (tmp_path / 'source.zip').write_bytes(b'modified')
    with pytest.raises(ValueError, match='corrupt'):
        fetch.acquire(source, tmp_path)


@pytest.mark.parametrize('payload', [b'short', b'too much data', b'altered!'])
def test_short_oversized_or_wrong_hash_is_not_published(tmp_path, payload):
    with patch.object(fetch.urllib.request, 'urlopen', return_value=Response(payload)):
        with pytest.raises(ValueError):
            fetch.acquire(item(), tmp_path)
    assert not (tmp_path / 'source.zip').exists()
    assert not list(tmp_path.glob('*.complete.json'))


def test_member_range_cannot_silently_download_archive(tmp_path):
    class NoBody(Response):
        def read(self, *_):
            pytest.fail('A server ignoring Range must be rejected before body reads')
    source = {**item(), 'range': [10, 17, 100]}
    with patch.object(fetch.urllib.request, 'urlopen', return_value=NoBody(b'')):
        with pytest.raises(ValueError, match='exact member range'):
            fetch.acquire(source, tmp_path)
    with patch.object(fetch.urllib.request, 'urlopen', return_value=Response(b'original', 206, {'Content-Range': 'bytes 10-17/100'})):
        assert fetch.acquire(source, tmp_path)['sha256'] == source['sha256']


def test_changed_item_and_receipt_are_rejected(tmp_path):
    source = item()
    with patch.object(fetch.urllib.request, 'urlopen', return_value=Response(b'original')):
        fetch.acquire(source, tmp_path)
    with pytest.raises(ValueError, match='item changed'):
        fetch.acquire({**source, 'url': 'https://example.invalid/new'}, tmp_path)
    receipt = tmp_path / 'source.zip.complete.json'
    data = json.loads(receipt.read_text())
    data['sha256'] = '0' * 64
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Receipt hash'):
        fetch.acquire(source, tmp_path)


def test_sql_literals_are_parsed_as_data_and_expressions_rejected():
    assert catalog.values("(1,'drone\\'s \\\\path',NULL,-1.5e2),(2,'bad')") == [[1, "drone's \\path", None, -150.0], [2, 'bad']]
    with pytest.raises(ValueError, match='Unsupported SQL literal'):
        catalog.values("(1,LOAD_FILE('/tmp/private'))")
    with pytest.raises(ValueError, match='Truncated'):
        catalog.values("(1,'unfinished")
    with pytest.raises(ValueError, match='column count'):
        catalog.parse_catalog("CREATE TABLE `x` (\n  `id` int\n) ENGINE=InnoDB;\nINSERT INTO `x` VALUES (1,2);\n")
