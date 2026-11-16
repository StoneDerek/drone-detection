#!/usr/bin/env python3
"""Build a verified local view of existing archives without changing originals."""

import argparse
import concurrent.futures
from pathlib import Path

from drone_detector.common import read_json, write_json
from drone_detector.corpus_intake import verify

ROOT = Path(__file__).resolve().parents[1]


def link_verified(item, source_root, output):
    if item.get("checksum_origin") != "publisher":
        raise ValueError("This view accepts only pinned publisher distribution digests")
    source = (Path(source_root) / item["source_id"] / item["filename"]).resolve()
    if not source.is_file():
        raise ValueError("Missing required original distribution: " + str(source))
    observed = verify(source, item)
    target = Path(output) / item["source_id"] / item["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if not target.is_symlink() or target.resolve() != source:
            raise ValueError("Existing archive view points elsewhere")
    else:
        target.symlink_to(source)
    receipt = dict(
        source_id=item["source_id"],
        filename=item["filename"],
        bytes=item["bytes"],
        **observed,
        publisher_checksum_verified=True,
        pinned_checksum_verified=True,
        verification="whole original file independently compared with pinned publisher digest; original not modified",
    )
    write_json(target.with_name(target.name + ".complete.json"), receipt)
    print("Verified original", item["source_id"], item["filename"], flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archives", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, choices=[1, 2], default=2)
    args = p.parse_args()
    if args.archives.resolve() == args.output.resolve():
        p.error("The source view must be separate from the original archives")
    manifest = read_json(ROOT / "data/corpus-intake-v3.json")
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        receipts = list(
            pool.map(
                lambda item: link_verified(item, args.archives, args.output), manifest["items"]
            )
        )
    write_json(
        args.output / "verification-complete.json",
        dict(
            files=len(receipts),
            bytes=sum(i["bytes"] for i in receipts),
            originals_modified=False,
            publisher_digests_verified=True,
        ),
    )


if __name__ == "__main__":
    main()
