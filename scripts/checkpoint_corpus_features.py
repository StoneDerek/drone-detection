#!/usr/bin/env python3
"""Pack completed numerical feature caches into verified Colab/Drive checkpoints."""

import argparse
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

from drone_detector.common import digest, file_hash, read_json, write_json


def validate(directory):
    index = read_json(directory / "index.json")
    encoder = read_json(directory / "encoder.json")
    members = {"index.json", "encoder.json"}
    for entry in index.values():
        if not re.fullmatch(r"[0-9a-f]{64}", entry["key"]):
            raise ValueError("Invalid content key in feature checkpoint")
        payload = directory / (entry["key"] + ".npz")
        receipt = payload.with_suffix(".json")
        meta = read_json(receipt)
        if digest(meta["specification"]) != entry["key"]:
            raise ValueError("Feature key differs from its specification")
        if file_hash(payload) != meta["sha256"]:
            raise ValueError("Corrupt feature payload")
        if meta["specification"]["encoder"] != encoder:
            raise ValueError("Mixed encoder identities in a feature checkpoint")
        if meta["specification"]["audio_sha256"] != entry["audio_sha256"]:
            raise ValueError("Feature/audio identity mismatch")
        members.update([payload.name, receipt.name])
    return index, members


def pack(directory, checkpoint):
    index, members = validate(directory)
    index_sha = file_hash(directory / "index.json")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="corpus-feature-checkpoint-") as temporary:
        archive = Path(temporary) / "features.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=1) as package:
            for name in sorted(members):
                package.add(directory / name, arcname=name, recursive=False)
        if file_hash(directory / "index.json") != index_sha:
            raise ValueError("Feature writer is still active; checkpoint only after it finishes")
        checksum = file_hash(archive)
        destination = checkpoint.with_name(checkpoint.name + ".partial")
        shutil.copyfile(archive, destination)
        if file_hash(destination) != checksum:
            raise ValueError("Checkpoint transfer read-back failed")
        destination.replace(checkpoint)
    receipt = {
        "schema_version": 1,
        "archive_sha256": checksum,
        "index_sha256": index_sha,
        "encoder_sha256": file_hash(directory / "encoder.json"),
        "records": len(index),
        "members": len(members),
        "scope": "Numerical frozen-feature cache; no audio, classification claims or new fitting",
    }
    write_json(checkpoint.with_name(checkpoint.name + ".complete.json"), receipt)
    print(json.dumps(receipt), flush=True)


def restore(directory, checkpoint):
    if (directory / "index.json").exists():
        index, _ = validate(directory)
        print(f"Keeping {len(index)} validated local cache records", flush=True)
        return
    receipt_path = checkpoint.with_name(checkpoint.name + ".complete.json")
    if not checkpoint.exists() and not receipt_path.exists():
        print("No earlier checkpoint; the full requested extraction will run", flush=True)
        return
    receipt = read_json(receipt_path)
    if receipt["schema_version"] != 1 or file_hash(checkpoint) != receipt["archive_sha256"]:
        raise ValueError("Checkpoint is missing or corrupt; preserve it for inspection")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="corpus-feature-restore-", dir=directory.parent
    ) as temporary:
        staging = Path(temporary)
        with tarfile.open(checkpoint, "r:gz") as package:
            for member in package.getmembers():
                if not member.isfile() or Path(member.name).name != member.name:
                    raise ValueError("Unexpected path in the numerical feature checkpoint")
            package.extractall(staging, filter="data")
        index, members = validate(staging)
        if (
            len(index) != receipt["records"]
            or len(members) != receipt["members"]
            or file_hash(staging / "index.json") != receipt["index_sha256"]
            or file_hash(staging / "encoder.json") != receipt["encoder_sha256"]
        ):
            raise ValueError("Checkpoint contents differ from its completion receipt")
        directory.mkdir(parents=True, exist_ok=True)
        for name in sorted(members - {"index.json"}):
            target = directory / name
            if target.exists() and file_hash(target) != file_hash(staging / name):
                raise ValueError("Local partial cache conflicts with the saved checkpoint")
            shutil.copyfile(staging / name, target)
        shutil.copyfile(staging / "index.json", directory / "index.json")
    print(f"Restored {len(index)} validated numerical feature records", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pack", "restore"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    (pack if args.stage == "pack" else restore)(args.directory, args.checkpoint)


if __name__ == "__main__":
    main()
