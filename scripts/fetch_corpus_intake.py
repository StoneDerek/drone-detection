#!/usr/bin/env python3
"""Fetch original distribution files; resume partials and verify pinned checksums."""

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drone_detector.corpus_intake import acquire, select_items  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus-intake-v3.json"))
    parser.add_argument("--profile", default="core")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-gib", type=float)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    items = select_items(manifest, args.profile)
    total = sum(item["bytes"] for item in items)
    print(json.dumps({"profile": args.profile, "files": len(items), "bytes": total,
                      "gib": total / 1024 ** 3,
                      "source_ids": sorted({item["source_id"] for item in items})}), flush=True)
    if args.plan_only:
        return
    if (args.budget_gib is None or not math.isfinite(args.budget_gib)
            or args.budget_gib <= 0 or total > args.budget_gib * 1024 ** 3):
        parser.error("Selected distribution files exceed the declared project allocation")
    args.output.mkdir(parents=True, exist_ok=True)
    needed = 0
    for item in items:
        path = args.output / item["source_id"] / item["filename"]
        partial = path.with_name(path.name + ".part")
        if not path.exists():
            needed += max(0, item["bytes"] - (partial.stat().st_size if partial.exists() else 0))
    if shutil.disk_usage(args.output).free < needed:
        parser.error("Filesystem has insufficient space for pending transfers")
    for item in items:
        result = acquire(item, args.output)
        print(json.dumps({key: result[key] for key in (
            "source_id", "filename", "sha256", "publisher_checksum_verified",
            "reused", "resumed_bytes")}), flush=True)
    print("Verified original distributions saved; no labels, splits or detector result assigned.")


if __name__ == "__main__":
    main()
