"""Read bounded members of the two-volume, non-ZIP64 FSD50K evaluation ZIP.

Offsets in a split ZIP are relative to the named disk, not a concatenated file.
The caller verifies whole-volume publisher checksums before constructing this
reader. Member size and CRC checks remain mandatory on every extraction.
"""

import bisect
import struct
import zlib
from pathlib import Path


class SplitZip:
    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        self.sizes = [p.stat().st_size for p in self.paths]
        self.bases = [0]
        for size in self.sizes:
            self.bases.append(self.bases[-1] + size)
        tail = self.read_at(max(0, self.bases[-1] - 65557), min(65557, self.bases[-1]))
        at = tail.rfind(b"PK\x05\x06")
        if at < 0 or len(tail) - at < 22:
            raise ValueError("Missing ZIP end record")
        _, disk, cd_disk, on_disk, count, length, offset, comment = struct.unpack_from(
            "<4s4H2IH", tail, at
        )
        if (
            disk != len(paths) - 1
            or cd_disk != disk
            or on_disk != count
            or count == 65535
            or offset == 0xFFFFFFFF
            or at + 22 + comment != len(tail)
        ):
            raise ValueError("Unsupported split ZIP layout / ZIP64")
        raw = self.read_at(self.bases[cd_disk] + offset, length)
        self.entries = {}
        pos = 0
        for _ in range(count):
            values = struct.unpack_from("<4s6H3I5H2I", raw, pos)
            if values[0] != b"PK\x01\x02":
                raise ValueError("Invalid central directory")
            flags, method, crc, packed, size = values[3], values[4], *values[7:10]
            n, extra, note, volume, local_offset = *values[10:14], values[16]
            name_bytes = raw[pos + 46 : pos + 46 + n]
            name = name_bytes.decode("utf-8" if flags & 2048 else "cp437")
            if name in self.entries or volume >= len(paths) or flags & 1 or method not in (0, 8):
                raise ValueError("Duplicate, encrypted or unsupported ZIP member")
            self.entries[name] = dict(
                name=name,
                flags=flags,
                method=method,
                crc32=crc,
                packed=packed,
                size=size,
                offset=self.bases[volume] + local_offset,
            )
            pos += 46 + n + extra + note
        if pos != len(raw):
            raise ValueError("Central directory length mismatch")

    def read_at(self, start, size):
        if start < 0 or size < 0 or start + size > self.bases[-1]:
            raise ValueError("Split ZIP read outside volumes")
        pieces = []
        while size:
            disk = bisect.bisect_right(self.bases, start) - 1
            offset = start - self.bases[disk]
            take = min(size, self.sizes[disk] - offset)
            with self.paths[disk].open("rb") as stream:
                stream.seek(offset)
                data = stream.read(take)
            if len(data) != take:
                raise ValueError("Truncated ZIP volume")
            pieces.append(data)
            start += take
            size -= take
        return b"".join(pieces)

    def read(self, name, limit=16 * 1024 * 1024):
        item = self.entries[name]
        if max(item["size"], item["packed"]) > limit:
            raise ValueError("ZIP member exceeds bounded extraction limit")
        header = self.read_at(item["offset"], 30)
        h = struct.unpack("<4s5H3I2H", header)
        if h[0] != b"PK\x03\x04" or h[2] != item["flags"] or h[3] != item["method"]:
            raise ValueError("Local and central ZIP headers disagree")
        raw_name = self.read_at(item["offset"] + 30, h[-2])
        if raw_name.decode("utf-8" if item["flags"] & 2048 else "cp437") != name:
            raise ValueError("Local ZIP name changed")
        packed = self.read_at(item["offset"] + 30 + h[-2] + h[-1], item["packed"])
        if item["method"] == 8:
            decoder = zlib.decompressobj(-15)
            data = decoder.decompress(packed, item["size"] + 1)
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError("Invalid or oversized deflate payload")
        else:
            data = packed
        if len(data) != item["size"] or zlib.crc32(data) != item["crc32"]:
            raise ValueError("ZIP member size / CRC mismatch")
        return data
