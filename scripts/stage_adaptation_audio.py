#!/usr/bin/env python3
"""Stage verified canonical inputs locally without changing waveform bytes."""

import argparse
import shutil
from pathlib import Path

from drone_detector.common import digest, file_hash, write_json, write_jsonl
from drone_detector.data import load_manifest


def stage(manifest, output):
    rows = load_manifest(manifest, require_split=True)
    output = Path(output).resolve()
    (output / "prepared/audio").mkdir(parents=True, exist_ok=True)
    staged = []
    for number, original in enumerate(rows, 1):
        row = dict(original)
        destination = output / "prepared/audio" / (digest(row["id"]) + ".wav")
        if destination.is_symlink():
            raise ValueError("Staged audio must be a regular local file")
        if destination.exists():
            if file_hash(destination) != row["sha256"]:
                raise ValueError("Existing staged input differs; preserve it for review")
        else:
            temporary = destination.with_suffix(".partial")
            shutil.copyfile(row["path"], temporary)
            if file_hash(temporary) != row["sha256"]:
                raise ValueError("Source bytes changed during local staging")
            temporary.replace(destination)
        row["path"] = str(destination)
        staged.append(row)
        if number % 500 == 0:
            print("Verified local inputs", number, "/", len(rows), flush=True)
    result = output / "prepared/manifest.jsonl"
    write_jsonl(result, staged)
    load_manifest(result, require_split=True)
    write_json(
        output / "prepared/preparation-complete.json",
        dict(
            manifest_sha256=file_hash(result),
            source_manifest_sha256=file_hash(manifest),
            rows=len(rows),
            transformation="binary copy only; canonical waveform bytes unchanged",
        ),
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(stage(args.manifest, args.output), flush=True)


if __name__ == "__main__":
    main()
