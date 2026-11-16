#!/usr/bin/env python3
"""Acquire checksum-pinned metadata and the two FSD50K evaluation volumes.

The original study inputs and their BEATs embeddings must already be present.
WSU and AuDroK bounded audio reads happen in prepare_confuser_study.py.
"""

import argparse
import concurrent.futures
import hashlib
import urllib.request
from pathlib import Path

from drone_detector.common import read_json, write_json
from drone_detector.corpus_intake import acquire

ROOT = Path(__file__).resolve().parents[1]


def document(item, work):
    path = work / item["path"]
    if path.exists():
        raw = path.read_bytes()
    else:
        with urllib.request.urlopen(item["url"], timeout=90) as response:
            raw = response.read(30_000_001)
    if len(raw) > 30_000_000:
        raise ValueError("Unexpectedly large metadata document")
    if item.get("bytes") and len(raw) != item["bytes"]:
        raise ValueError("Metadata size changed")
    for algo in ("sha256", "md5"):
        if item.get(algo) and hashlib.new(algo, raw).hexdigest() != item[algo]:
            raise ValueError("Pinned metadata checksum mismatch: " + item["path"])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(raw)
        temporary.replace(path)
    print("Verified metadata", item["path"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=Path("work/confuser-study-v1"))
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    lock = read_json(ROOT / "data/confuser-study-v1-sources.json")
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda item: document(item, args.work), lock["documents"]))
    for filename, data in [
        ("fsd-acquisition-plan.json", lock["fsd_audio"]),
        ("audrok-archives.json", lock["audrok_public_files"]),
    ]:
        target = args.work / filename
        if target.exists() and read_json(target) != data:
            raise ValueError("Acquisition plan changed: " + filename)
        write_json(target, data)
    if not args.metadata_only:
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            for receipt in pool.map(
                lambda item: acquire(item, args.work / "archives"), lock["fsd_audio"]
            ):
                print(receipt["filename"], "publisher checksum verified", flush=True)


if __name__ == "__main__":
    main()
