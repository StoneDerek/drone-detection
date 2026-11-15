#!/usr/bin/env python3
"""Fetch pinned official BEATs code and an explicitly identified checkpoint mirror.

No inference API fetches executable assets automatically. Interrupted downloads
are restarted; completed assets are rehashed and reused. Existing changed files
are preserved and rejected rather than silently overwritten.
"""

import argparse
import urllib.request
from pathlib import Path

from drone_detector.beats_encoder import (
    CHECKPOINT_BYTES, CHECKPOINT_SHA256, CHECKPOINT_URL, CODE_COMMIT,
    MIRROR_REVISION, MODEL_NAME, OFFICIAL_CHECKPOINT_LINK, SOURCE_ASSETS,
    source_url, verify_assets,
)
from drone_detector.common import file_hash, write_json


def fetch(url, target, size, checksum):
    if target.exists():
        if target.stat().st_size != size or file_hash(target) != checksum:
            raise ValueError(f"Existing asset differs from pinned content: {target}")
        return
    partial = target.with_suffix(target.suffix + ".partial")
    count = 0
    request = urllib.request.Request(url, headers={"User-Agent": "drone-detection-beats-intake/1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            count += len(block)
            if count > size:
                raise ValueError(f"Oversized asset: {target.name}")
            stream.write(block)
    if count != size or file_hash(partial) != checksum:
        raise ValueError(f"Incomplete or changed asset: {target.name}")
    partial.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("work/beats-encoder"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, (size, checksum) in SOURCE_ASSETS.items():
        fetch(source_url(name), args.output / name, size, checksum)
    fetch(CHECKPOINT_URL, args.output / "model.pt", CHECKPOINT_BYTES, CHECKPOINT_SHA256)
    manifest = {
        "schema_version": 1, "model": MODEL_NAME,
        "official_code_repository": "https://github.com/microsoft/unilm/tree/" + CODE_COMMIT + "/beats",
        "code_commit": CODE_COMMIT, "code_license": "MIT (included LICENSE)",
        "official_checkpoint_link": OFFICIAL_CHECKPOINT_LINK,
        "checkpoint_download": CHECKPOINT_URL,
        "checkpoint_mirror_revision": MIRROR_REVISION,
        "checkpoint_provenance_limit": (
            "Pinned third-party mirror of named published checkpoint. "
            "Not independently byte-compared with official OneDrive download. "
            "Official README does not specify separate checkpoint terms."
        ),
        "sha256": {**{name: value[1] for name, value in SOURCE_ASSETS.items()},
                   "model.pt": CHECKPOINT_SHA256},
    }
    write_json(args.output / "assets.json", manifest)
    verify_assets(args.output)
    print(f"Verified {MODEL_NAME} assets in {args.output}; inference remains offline.")


if __name__ == "__main__":
    main()
