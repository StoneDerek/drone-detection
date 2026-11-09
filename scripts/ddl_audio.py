#!/usr/bin/env python3
"""Read published real DDL audio via verified HTTPS byte ranges (no synthetic set).

ZIP member CRCs are checked by zipfile; member SHA-256 is recorded. The publisher's
whole-archive MD5 is recorded but cannot be claimed verified for a partial fetch.
"""

import argparse
import collections
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/6459183/files/MLSP_2022_Real_Data.zip?download=1"
LENGTH = 12628837686


class RangeReader(io.RawIOBase):
    def __init__(self, url=URL, length=LENGTH, block_size=4 * 1024 * 1024, cache_dir=None):
        self.url, self.length, self.block_size = url, length, block_size
        self.position, self.transferred = 0, 0
        self.cache = collections.OrderedDict()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset
        if position < 0:
            raise ValueError("Negative remote ZIP seek")
        self.position = position
        return position

    def read(self, size=-1):
        size = min(self.length - self.position, size if size >= 0 else self.length - self.position)
        if size > 256 * 1024 * 1024:
            raise ValueError("Refusing an unbounded range read")
        pieces = []
        while size > 0:
            index = self.position // self.block_size
            if index not in self.cache:
                start = index * self.block_size
                end = min(self.length - 1, start + self.block_size - 1)
                disk = self.cache_dir / f"{index}.bin" if self.cache_dir else None
                checksum = disk.with_suffix('.sha256') if disk else None
                if disk and disk.exists() and checksum.exists():
                    data = disk.read_bytes()
                    if len(data) != end - start + 1 or hashlib.sha256(data).hexdigest() != checksum.read_text().strip():
                        raise ValueError("Cached ZIP block changed")
                    self.cache[index] = data
                    while len(self.cache) > 8:
                        self.cache.popitem(last=False)
                    continue
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}",
                                                              "Accept-Encoding": "identity"})
                with urllib.request.urlopen(req, timeout=120) as response:
                    expected = f"bytes {start}-{end}/{self.length}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor the exact requested ZIP byte range")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise ValueError("Truncated remote ZIP block")
                self.transferred += len(data)
                self.cache[index] = data
                if disk:
                    disk.write_bytes(data)
                    checksum.write_text(hashlib.sha256(data).hexdigest() + '\n')
                while len(self.cache) > 8:
                    self.cache.popitem(last=False)
            data = self.cache[index]
            offset = self.position - index * self.block_size
            take = min(size, len(data) - offset)
            pieces.append(data[offset:offset + take])
            self.position += take
            size -= take
        return b"".join(pieces)


def inventory(output):
    reader = RangeReader()
    with zipfile.ZipFile(reader) as archive:
        entries = [{"path": r.filename, "size": r.file_size, "compressed_size": r.compress_size,
                    "crc32": r.CRC, "header_offset": r.header_offset}
                   for r in archive.infolist() if not r.is_dir()]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"source": URL, "length": LENGTH,
        "publisher_archive_md5": "4a6d4da4e1c732550c1ccd8d29dd16f8", "whole_archive_md5_verified": False,
        "metadata_bytes_transferred": reader.transferred, "entries": entries}, indent=2) + "\n")
    print("Archive entries", len(entries), "metadata bytes transferred", reader.transferred)
    print("Examples", [(r["path"], r["size"]) for r in entries[:8]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="runs/real-v2/ddl-inventory.json")
    args = parser.parse_args()
    inventory(args.inventory)
