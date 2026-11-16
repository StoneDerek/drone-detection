#!/usr/bin/env python3
"""Recover the FPV audit inputs from the complete, publisher-verified ZIP."""

import argparse
import hashlib
import zipfile
from pathlib import Path

from drone_detector.common import write_json
from drone_detector.corpus_intake import verify


def prepare(archive_path, output):
    if any((output / name).exists() for name in ['Fpv.zip.directory.json', 'annotation-manifest.json']):
        raise ValueError('Existing metadata manifest; preserve it and use a new input directory')
    observed = verify(archive_path, dict(bytes=2152999558, md5="b06064de415a1d8fdf16b3e508ba96a0"))
    entries, annotations = [], []
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            entries.append(dict(path=item.filename, bytes=item.file_size,
                                compressed_bytes=item.compress_size, crc32=item.CRC,
                                header_offset=item.header_offset, compression=item.compress_type))
            if not item.filename.lower().endswith('.txt'):
                continue
            if item.file_size > 2 * 1024 * 1024:
                raise ValueError('Unreviewed annotation size')
            payload = archive.read(item)
            local_name = hashlib.sha256(('Fpv.zip::' + item.filename).encode()).hexdigest()[:20] + '.txt'
            target = output / 'field-annotations' / local_name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != payload:
                raise ValueError('Existing annotation differs; use an isolated input directory')
            target.write_bytes(payload)
            annotations.append(dict(archive='Fpv.zip', member=item.filename, bytes=len(payload),
                                    crc32=item.CRC, sha256=hashlib.sha256(payload).hexdigest(),
                                    local_name=local_name))
    for name, value in [('Fpv.zip.directory.json', entries), ('annotation-manifest.json', annotations)]:
        write_json(output / name, value)
    write_json(output / 'input-preparation.json', dict(archive=observed, annotations=len(annotations)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.archive, args.output)


if __name__ == '__main__':
    main()
