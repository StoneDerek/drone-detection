#!/usr/bin/env python3
"""Fetch the immutable official encoder, with its recorded archive checksum."""

import argparse
import hashlib
import io
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = 'https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed'
SHA256 = 'b80da2a1a56926fb0767205051a200dd7b3beaf3ea1ea126c42a53943996e5e0'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='runs/real-v1/encoders/yamnet')
    args = p.parse_args()
    output = Path(args.output)
    cached = Path('runs/real-v1/downloads/yamnet.tar.gz')
    if cached.exists():
        payload = cached.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=120) as response:
            payload = response.read()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError('Encoder archive changed; inspect the source rather than replacing its expected checksum')
    cached.parent.mkdir(parents=True, exist_ok=True)
    if not cached.exists():
        cached.write_bytes(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='verified-yamnet-') as temporary:
        root = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            for member in archive.getmembers():
                path = (root / member.name).resolve()
                if not path.is_relative_to(root.resolve()) or not (member.isfile() or member.isdir()):
                    raise ValueError('Unexpected encoder archive member')
            archive.extractall(root, filter='data')
        if output.exists():
            for source in root.rglob('*'):
                if source.is_file():
                    target = output / source.relative_to(root)
                    if not target.exists() or hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
                        raise ValueError('Existing encoder differs; preserving it for inspection')
        else:
            shutil.copytree(root, output)
    print('Official YAMNet archive and extracted encoder verified.')


if __name__ == '__main__':
    main()
