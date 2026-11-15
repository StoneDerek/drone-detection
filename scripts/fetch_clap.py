#!/usr/bin/env python3
"""Fetch the immutable official CLAP release; inference never accesses network."""

import argparse
import urllib.request
from pathlib import Path

from drone_detector.clap_encoder import ASSETS, MODEL_NAME, MODEL_REVISION, asset_url, verify_assets
from drone_detector.common import file_hash, write_json


def fetch(url, target, size, checksum):
    if target.exists():
        if target.stat().st_size != size or file_hash(target) != checksum:
            raise ValueError(f'Existing asset differs from pinned content: {target}')
        return
    partial = target.with_suffix(target.suffix + '.partial')
    count = 0
    request = urllib.request.Request(url, headers={'User-Agent': 'drone-detection-clap-intake/1'})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as stream:
        for block in iter(lambda: response.read(1024 * 1024), b''):
            count += len(block)
            if count > size:
                raise ValueError(f'Oversized asset: {target.name}')
            stream.write(block)
    if count != size or file_hash(partial) != checksum:
        raise ValueError(f'Incomplete or changed asset: {target.name}')
    partial.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('work/clap-encoder'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, (size, checksum) in ASSETS.items():
        fetch(asset_url(name), args.output / name, size, checksum)
    write_json(args.output / 'assets.json', {
        'schema_version': 1, 'model': MODEL_NAME, 'revision': MODEL_REVISION,
        'source': f'https://huggingface.co/{MODEL_NAME}/tree/{MODEL_REVISION}',
        'license': 'Apache-2.0 per included official model card',
        'checkpoint_format': 'PyTorch state_dict; loaded only with torch weights_only=True',
        'sha256': {name: value[1] for name, value in ASSETS.items()},
    })
    verify_assets(args.output)
    print(f'Verified {MODEL_NAME} assets in {args.output}; inference remains offline.')


if __name__ == '__main__':
    main()
