import importlib.util
import struct
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('candidate_headers', ROOT / 'scripts/audit_dronenoise_headers.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def prefix(encoding, channels, bits, frames):
    block = channels * (bits // 8)
    fmt = struct.pack('<HHIIHH', encoding, channels, 50000, 50000 * block, block, bits)
    chunks = b'fmt ' + struct.pack('<I', len(fmt)) + fmt
    # Odd-sized metadata chunk exercises RIFF's padding rule.
    chunks += b'JUNK' + struct.pack('<I', 3) + b'abc\x00'
    chunks += b'data' + struct.pack('<I', frames * block)
    payload = b'RIFF' + struct.pack('<I', 4 + len(chunks) + frames * block) + b'WAVE' + chunks
    return payload, len(payload) + frames * block


@pytest.mark.parametrize('encoding,channels,bits', [(1, 2, 16), (3, 1, 32)])
def test_pcm_and_float_headers_count_frames_not_channels(encoding, channels, bits):
    header, length = prefix(encoding, channels, bits, 1000000)
    result = audit.wav_header(header, length)
    assert result['duration_seconds'] == 20
    assert result['channels'] == channels
    with pytest.raises(ValueError, match='extent'):
        audit.wav_header(header, length - 1)


def test_ignored_http_range_is_rejected_without_reading_body(tmp_path):
    class FullResponse:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, *_):
            pytest.fail('Do not read a full-file response when a range is required')

    with patch.object(audit.urllib.request, 'urlopen', return_value=FullResponse()):
        with pytest.raises(ValueError, match='exact bounded range'):
            audit.audit_member({'size': 4000080, 'download_url': 'https://example.invalid/file'}, tmp_path)


def test_incomplete_native_frame_remains_a_visible_diagnostic():
    header, length = prefix(1, 6, 32, 100)
    malformed = bytearray(header)
    struct.pack_into('<I', malformed, 4, length)
    struct.pack_into('<I', malformed, len(header) - 4, 100 * 24 + 8)
    with pytest.raises(ValueError, match='incomplete interleaved frame'):
        audit.wav_header(malformed, length + 8)
    facts = audit.wav_header(malformed, length + 8, allow_partial_frame=True)
    assert facts['complete_frames'] == 100 and facts['partial_frame_bytes'] == 8
    assert facts['duration_seconds'] == 100 / 50000
