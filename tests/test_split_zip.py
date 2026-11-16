import io
import struct
import zipfile
import zlib

import pytest

from drone_detector.split_zip import SplitZip


def volumes(tmp_path):
    """Two members, one crossing the disk boundary, disk-relative directory."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("first.wav", bytes(range(256)) * 20)
        z.writestr("second.wav", b"second member")
    raw = bytearray(buffer.getvalue())
    eocd = raw.rfind(b"PK\x05\x06")
    count = struct.unpack_from("<H", raw, eocd + 10)[0]
    cd_offset = struct.unpack_from("<I", raw, eocd + 16)[0]
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        boundary = z.getinfo("second.wav").header_offset - 5
    pos = cd_offset
    for _ in range(count):
        offset = struct.unpack_from("<I", raw, pos + 42)[0]
        if offset >= boundary:
            struct.pack_into("<H", raw, pos + 34, 1)
            struct.pack_into("<I", raw, pos + 42, offset - boundary)
        n, extra, comment = struct.unpack_from("<3H", raw, pos + 28)
        pos += 46 + n + extra + comment
    struct.pack_into("<2H", raw, eocd + 4, 1, 1)
    struct.pack_into("<I", raw, eocd + 16, cd_offset - boundary)
    a, b = tmp_path / "audio.z01", tmp_path / "audio.zip"
    a.write_bytes(raw[:boundary])
    b.write_bytes(raw[boundary:])
    return a, b


def test_members_and_payload_crossing_disk(tmp_path):
    reader = SplitZip(volumes(tmp_path))
    assert reader.read("first.wav") == bytes(range(256)) * 20
    assert reader.read("second.wav") == b"second member"
    assert reader.read_at(reader.sizes[0] - 3, 9)
    with pytest.raises(ValueError, match="bounded"):
        reader.read("first.wav", limit=10)


def test_crc_failure_and_range_bounds(tmp_path):
    reader = SplitZip(volumes(tmp_path))
    reader.entries["first.wav"]["crc32"] = zlib.crc32(b"wrong")
    with pytest.raises(ValueError, match="CRC"):
        reader.read("first.wav")
    with pytest.raises(ValueError, match="outside"):
        reader.read_at(reader.bases[-1], 1)


def test_local_filename_mismatch(tmp_path):
    paths = volumes(tmp_path)
    raw = bytearray(paths[0].read_bytes())
    raw[30] = ord("x")
    paths[0].write_bytes(raw)
    reader = SplitZip(paths)
    with pytest.raises(ValueError, match="name"):
        reader.read("first.wav")
