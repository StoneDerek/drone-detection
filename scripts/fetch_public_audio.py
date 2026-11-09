#!/usr/bin/env python3
"""Fetch versioned original public audio, preserving source metadata and hashes.

No authentication or license assumptions are inferred from mirror titles. The
Kaggle D3Set mirror is explicitly linked by the primary IntRoLab repository.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path


def get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "acoustic-drone-research/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def fetch_blob(item, root):
    path = root / item["path"]
    if path.exists():
        data = path.read_bytes()
    else:
        data = get(item["url"])
    if item.get("git_blob"):
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if actual != item["git_blob"]:
            raise ValueError(f"Git content hash mismatch: {item['path']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)
    return {**item, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/real-v1/public")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / "metadata"
    metadata.mkdir(exist_ok=True)
    tasks, sources = [], []
    for name, repo, commit, license_name in [
        ("svanstrom", "DroneDetectionThesis/Drone-detection-dataset", "442d0c6750708cc221636f7a1fd31cbd8cb0aeb8", "CC0-1.0"),
        ("esc10", "karolpiczak/ESC-50", "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6", "CC-BY (ESC-10 subset only)"),
    ]:
        tree = json.loads(get(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1"))
        if tree.get("truncated"):
            raise ValueError("Incomplete source inventory")
        base = f"https://raw.githubusercontent.com/{repo}/{commit}/"
        (metadata / f"{name}-tree.json").write_text(json.dumps(tree, indent=2))
        source = {"name": name, "repository": repo, "commit": commit, "license": license_name}
        if name == "esc10":
            raw = get(base + "meta/esc50.csv")
            (metadata / "esc50.csv").write_bytes(raw)
            selected = {row["filename"] for row in csv.DictReader(io.StringIO(raw.decode()))
                        if row["esc10"].lower() == "true"}
            if len(selected) != 400:
                raise ValueError("Unexpected ESC-10 membership")
        for item in tree["tree"]:
            path = item["path"]
            wanted = (name == "svanstrom" and path.startswith("Data/Audio/") and path.endswith(".wav"))
            wanted |= (name == "esc10" and path.startswith("audio/") and Path(path).name in selected)
            if wanted:
                tasks.append({"source": name, "version": commit,
                              "path": f"{name}/{path}", "url": base + urllib.parse.quote(path),
                              "git_blob": item["sha"], "license": license_name})
        for filename in ["README.md", "LICENSE"]:
            (metadata / f"{name}-{filename.replace('/', '_')}").write_bytes(get(base + filename))
        sources.append(source)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for index, result in enumerate(pool.map(lambda task: fetch_blob(task, output), tasks), 1):
            results.append(result)
            if index % 50 == 0:
                print(f"Verified {index}/{len(tasks)} original audio files", flush=True)
    (output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    (output / "downloads.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in results))
    print(f"Downloaded and content-verified {len(results)} real-audio files", flush=True)


if __name__ == "__main__":
    main()
