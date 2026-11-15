#!/usr/bin/env python3
"""Parallelize unchanged per-crop extraction with disjoint inputs and safe merging.

Each child uses the same pinned extractor, encoder identity and one-thread native
inference. Sharding changes scheduling, never crops, representations or models.
Canonical output must have no other active writer for this encoder family.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from drone_detector.common import digest, file_hash, read_json, write_json
from drone_detector.data import load_manifest

ROOT = Path(__file__).resolve().parents[1]


def publish_json(path, data):
    temporary = path.with_name(path.name + ".shard-partial")
    write_json(temporary, data)
    temporary.replace(path)


def link_checked(source, destination):
    if destination.exists():
        if file_hash(source) != file_hash(destination):
            raise ValueError("Conflicting feature payload")
    else:
        os.link(source, destination)


def run(args):
    extractor = ROOT / "scripts/extract_corpus_study_features.py"
    rows = [
        r
        for r in load_manifest(args.prepared / "manifest.jsonl", require_split=True)
        if r["split"] in args.splits
    ]
    if not rows:
        raise ValueError("Empty extraction scope")
    if "final" in args.splits and not (args.prepared / "final-evaluation-authorized.json").exists():
        raise ValueError("Final extraction is not authorized")
    plan = {
        "manifest_sha256": file_hash(args.prepared / "manifest.jsonl"),
        "extractor_sha256": file_hash(extractor),
        "family": args.family,
        "splits": sorted(args.splits),
        "shards": args.shards,
        "threads_per_child": 1,
        "device": args.device,
        "assignment": "sha256-record-id modulo shard count; all accepted inputs retained",
    }
    workspace = args.workspace / digest(plan)[:20]
    workspace.mkdir(parents=True, exist_ok=True)
    write_json(workspace / "plan.json", plan)
    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "index.json"
    previous = read_json(index_path) if index_path.exists() else {}
    expected = []
    folders = []
    for number in range(args.shards):
        assigned = [r for r in rows if int(digest(r["id"]), 16) % args.shards == number]
        identifiers = {r["id"] for r in assigned}
        expected.append(identifiers)
        folder = workspace / str(number)
        prepared = folder / "prepared"
        output = folder / "features"
        (prepared / "parts").mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        write_json(prepared / "parts/accepted.json", {"records": assigned, "plan": plan})
        if "final" in args.splits:
            write_json(
                prepared / "final-evaluation-authorized.json",
                read_json(args.prepared / "final-evaluation-authorized.json"),
            )
        shard_index = read_json(output / "index.json") if (output / "index.json").exists() else {}
        if set(shard_index) - identifiers:
            raise ValueError("Shard cache has unexpected records")
        for identifier in identifiers & set(previous):
            entry = previous[identifier]
            if identifier in shard_index and shard_index[identifier] != entry:
                raise ValueError("Canonical and shard feature identities disagree")
            for suffix in [".npz", ".json"]:
                name = entry["key"] + suffix
                link_checked(args.output / name, output / name)
            shard_index[identifier] = entry
        publish_json(output / "index.json", shard_index)
        folders.append((prepared, output, folder / "extraction.log"))
    print(
        json.dumps(
            {
                "plan": plan,
                "records": len(rows),
                "cached": sum(r["id"] in previous for r in rows),
                "shard_sizes": [len(ids) for ids in expected],
            }
        ),
        flush=True,
    )
    if args.plan_only:
        return
    workers = []
    handles = []
    try:
        for prepared, output, logfile in folders:
            command = [
                args.python,
                extractor,
                "--prepared",
                prepared,
                "--output",
                output,
                "--family",
                args.family,
                "--threads",
                "1",
                "--device",
                args.device,
                "--splits",
                *args.splits,
            ]
            if args.family != "spectral":
                command += ["--assets", args.assets]
            handle = logfile.open("a")
            handles.append(handle)
            workers.append(
                subprocess.Popen(
                    [str(v) for v in command], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT
                )
            )
        while any(worker.poll() is None for worker in workers):
            if any(worker.poll() not in {None, 0} for worker in workers):
                raise RuntimeError("A shard failed; inspect its extraction log")
            time.sleep(10)
        if any(worker.returncode != 0 for worker in workers):
            raise RuntimeError("A shard failed; inspect its extraction log")
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for handle in handles:
            handle.close()
    if file_hash(extractor) != plan["extractor_sha256"]:
        raise ValueError("Extractor source changed during execution")
    merged = read_json(index_path) if index_path.exists() else {}
    for identifiers, (_, output, _) in zip(expected, folders, strict=True):
        index = read_json(output / "index.json")
        if set(index) != identifiers:
            raise ValueError("Shard did not retain every assigned record")
        identity = read_json(output / "encoder.json")
        if (args.output / "encoder.json").exists():
            if read_json(args.output / "encoder.json") != identity:
                raise ValueError("Shard encoder differs from the canonical encoder")
        else:
            write_json(args.output / "encoder.json", identity)
        for identifier, entry in index.items():
            if identifier in merged and merged[identifier] != entry:
                raise ValueError("Refusing to replace an existing canonical feature")
            meta = read_json(output / (entry["key"] + ".json"))
            payload = output / (entry["key"] + ".npz")
            if file_hash(payload) != meta["sha256"] or meta["specification"]["encoder"] != identity:
                raise ValueError("Invalid shard feature receipt")
            if meta["specification"]["audio_sha256"] != entry["audio_sha256"]:
                raise ValueError("Shard receipt/audio identity mismatch")
            for suffix in [".npz", ".json"]:
                name = entry["key"] + suffix
                link_checked(output / name, args.output / name)
            merged[identifier] = entry
    publish_json(index_path, merged)
    write_json(
        workspace / "complete.json",
        {"plan": plan, "records": len(rows), "index_sha256": file_hash(index_path)},
    )
    print(json.dumps({"merged": args.family, "records": len(rows)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--family", choices=["spectral", "beats", "clap"], required=True)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--splits", nargs="+", default=["train", "development"])
    parser.add_argument("--shards", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--plan-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
