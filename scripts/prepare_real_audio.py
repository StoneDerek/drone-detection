#!/usr/bin/env python3
"""Prepare real publisher-labeled clips, preserving conservative source/day splits.

No artificial event boundaries are assigned. Weak publisher labels remain weak.
D3Set's ground-array channel is selected explicitly; onboard recordings and
overlapping Delivery/SNR drone derivatives are excluded from learning/testing.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from drone_detector.common import file_hash, write_json, write_jsonl
from drone_detector.data import audit, components, load_manifest


def get(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--public", default="runs/real-v1/public")
    p.add_argument("--archive", default="runs/real-v1/downloads/kaggle-d3set-v1.zip")
    p.add_argument("--output", default="runs/real-v1/corpus")
    args = p.parse_args()
    output, public = Path(args.output), Path(args.public)
    if output.exists():
        raise ValueError("Use a fresh corpus directory")
    (output / "audio").mkdir(parents=True)
    annotations = []
    esc = {r["filename"]: r for r in csv.DictReader((public / "metadata/esc50.csv").open())}
    for download in [json.loads(line) for line in (public / "downloads.jsonl").read_text().splitlines()]:
        path = public / download["path"]
        name, source = path.name, download["source"]
        if source == "svanstrom":
            label = "drone" if name.startswith("DRONE_") else "non_drone"
            split, session, original = "final", "unknown-originals-entire-source", "unknown-originals-entire-source"
            kind = name.split("_")[0].lower() if label == "non_drone" else None
        else:
            item = esc[name]
            label, kind = "non_drone", item["category"]
            split = "train" if int(item["fold"]) <= 3 else ("development" if item["fold"] == "4" else "final")
            session = original = "freesound-" + str(item["src_file"])
        identity = source + "-" + path.stem
        relative = f"audio/{identity}.wav"
        shutil.copyfile(path, output / relative)
        annotations.append({"id": identity, "path": relative, "source": source,
            "source_version": download["version"], "session_id": session, "original_id": original,
            "origin": "authentic", "label": label, "annotation_status": "weak",
            "label_evidence": "Publisher clip-level class; no independent full temporal annotation",
            "events": [], "ambiguous": [], "license": download["license"],
            "negative_kind": kind, "split_choice": split,
            "provenance_review": "Pinned publisher Git content verified; original Freesound IDs kept together for ESC, entire Svanstrom source held out because original session mapping is unknown",
            "source_recording": {"url": download["url"], "sha256": download["sha256"]}})
    d3_tree = json.loads(get("https://api.github.com/repos/introlab/d3set/git/trees/b7f80cfbb1393ed490a8677cb1550ed95b076718?recursive=1"))
    commit = d3_tree["sha"]
    if d3_tree.get("truncated"):
        raise ValueError("Incomplete primary D3Set tree")
    primary = {r["path"]: r for r in d3_tree["tree"]}
    archive_sha = file_hash(args.archive)
    if archive_sha != "955f0010ec37f0c4b580e75a95b6e5782a676763277211a0d328ac6d32f9c860":
        raise ValueError("Kaggle D3Set v1 archive hash changed; inspect upstream version before proceeding")
    with zipfile.ZipFile(args.archive) as z:
        chosen = [n for n in z.namelist() if
                  ("/Flights/" in n and n.endswith("/matrix.wav") and "/SNR test/" not in n)
                  or ("/Noise/" in n and n.lower().endswith(".wav"))]
        # Negative-only SNR excerpt is used for training, never as new independent
        # evidence. The drone SNR excerpt is excluded as a shared-flight derivative.
        chosen += [n for n in z.namelist() if "/SNR test/background noise/" in n and n.endswith("matrix.wav")]
        for index, name in enumerate(chosen):
            source_path = name.split("/", 1)[1]
            primary_info = primary[source_path]
            raw_url = f"https://raw.githubusercontent.com/introlab/d3set/{commit}/" + urllib.parse.quote(source_path)
            descriptor = get(raw_url)
            payload = z.read(name)
            if descriptor.startswith(b"version https://git-lfs.github.com/spec/v1"):
                expected = re.search(rb"oid sha256:([0-9a-f]{64})", descriptor).group(1).decode()
                if hashlib.sha256(payload).hexdigest() != expected:
                    raise ValueError(f"Kaggle mirror differs from pinned primary D3Set: {source_path}")
            else:
                git_hash = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
                if git_hash != primary_info["sha"]:
                    raise ValueError("Primary Git hash mismatch")
            samples, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
            if "/Noise/" in name:
                label = "non_drone"
                original = "pixabay-" + source_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                session = original
                value = int(hashlib.sha256(original.encode()).hexdigest()[:8], 16) % 10
                split = "train" if value < 6 else "development" if value < 8 else "final"
                kind = name.split("/")[-2]
            elif "/background noise/" in name:
                label, split = "non_drone", "train"
                session = original = "unknown-background-parent-training-only"
                kind = "same-array-background"
            else:
                label, kind = "drone", None
                original = name.split("/")[-2]
                normalized = original.lower().replace("_", " ")
                day = re.match(r"(\d+)\s*(\w+)", normalized)
                if day is None:
                    raise ValueError("Unrecognized flight day; refuse to split by arbitrary file")
                session = "flight-day-" + day[1] + "-" + day[2]
                if day[2] in {"avril", "mai"}:
                    split = "final"
                elif day[1] == "23" and day[2] == "oct":
                    split = "development"
                else:
                    split = "train"
            # Select one physical array channel. Do not average a microphone
            # array into a virtual microphone, or use onboard motor recordings.
            nonzero = np.sqrt(np.mean(samples.astype(np.float64)**2, axis=0)) > 1e-6
            valid = np.flatnonzero(nonzero)
            if not len(valid):
                continue
            channel = int(valid[0])
            if samples.shape[1] == 1:
                channel = 0
            identity = "d3-" + hashlib.sha256(source_path.encode()).hexdigest()[:16]
            relative = f"audio/{identity}.wav"
            sf.write(output / relative, samples[:, channel], rate, subtype="PCM_16")
            annotations.append({"id": identity, "path": relative, "source": "d3set-ground" if label == "drone" or kind == "same-array-background" else "d3set-noise",
                "source_version": "kaggle-v1-primary-" + commit,
                "session_id": session, "original_id": original, "origin": "authentic",
                "label": label, "annotation_status": "weak", "events": [], "ambiguous": [],
                "label_evidence": "Primary publisher flight/absence class; no audibility or temporal-boundary confirmation",
                "license": "GPL-3.0 repository; imported noise retains Pixabay terms; raw audio not redistributed",
                "negative_kind": kind, "split_choice": split,
                "device": "D3Set ground array, one physical channel" if samples.shape[1] > 1 else None,
                "environment": "Publisher outdoor flights" if label == "drone" else None,
                "provenance_review": "Primary Git/LFS content verified against Kaggle bytes. Same flight days kept together; Delivery and SNR drone derivatives excluded. No per-flight independence claim.",
                "source_recording": {"url": raw_url, "sha256": hashlib.sha256(payload).hexdigest(),
                    "archive_sha256": archive_sha, "original_channels": samples.shape[1],
                    "selected_channel": channel, "original_path": source_path}})
            if index % 10 == 0:
                print(f"Prepared D3Set real audio {index + 1}/{len(chosen)}", flush=True)
    # Independent field source; only entries in its published registry are used.
    batear_commit = "d0d6e67d9e7d167be9e0db249ef6a44acb52f732"
    base = f"https://raw.githubusercontent.com/batear-io/batear-datasets/{batear_commit}/"
    registry = json.loads(get(base + "metadata/samples.json"))["samples"]

    def field_sample(row):
        path = row["filename"]
        pointer = get(base + urllib.parse.quote(path))
        if pointer.startswith(b"version https://git-lfs.github.com/spec/v1"):
            expected = re.search(rb"oid sha256:([0-9a-f]{64})", pointer).group(1).decode()
            payload = get(f"https://media.githubusercontent.com/media/batear-io/batear-datasets/{batear_commit}/" + urllib.parse.quote(path))
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError("Batear LFS content mismatch")
        else:
            payload = pointer
        identity = "batear-" + Path(path).stem
        relative = "audio/" + identity + ".wav"
        (output / relative).write_bytes(payload)
        label = "non_drone" if row["category"] == "ambient" else "drone"
        date = row["timestamp_utc"][:10]
        return {"id": identity, "path": relative, "source": "batear-field", "source_version": batear_commit,
            "session_id": date, "original_id": identity, "origin": "authentic", "label": label,
            "annotation_status": "weak", "label_evidence": "Publisher field directory/registry label, no independent temporal annotation",
            "events": [], "ambiguous": [], "license": "MIT", "split_choice": "final",
            "negative_kind": "rural" if label == "non_drone" else None,
            "provenance_review": "Primary registry and LFS digest verified. All same-day FPV clips grouped because overlap/original mapping is uncertain. Entire source held out.",
            "source_recording": {"url": base + urllib.parse.quote(path), "sha256": hashlib.sha256(payload).hexdigest()}}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        annotations.extend(pool.map(field_sample, registry))
    write_jsonl(output / "annotations.jsonl", annotations)
    report = audit(output, output / "inventory-v1.jsonl", output / "annotations.jsonl")
    records = load_manifest(output / "inventory-v1.jsonl")
    choices = {r["id"]: r["split_choice"] for r in annotations}
    for row in records:
        row["split"] = choices[row["id"]]
    # Exact copies discovered across sources are conservatively retained in the
    # latest split; no descendant can straddle train/development/final.
    priority = {"train": 0, "development": 1, "final": 2}
    for group in set(components(records)):
        members = [r for r in records if r["group_id"] == group]
        split = max((r["split"] for r in members), key=priority.get)
        for row in members:
            row["split"] = split
    write_jsonl(output / "manifest-v1.jsonl", records)
    load_manifest(output / "manifest-v1.jsonl", True)
    write_json(output / "splits.json", {"manifest_sha256": file_hash(output / "manifest-v1.jsonl"),
        "method": "D3Set ground flight-day holdouts; no onboard/Delivery/SNR drone copies; ESC official parent-group folds; whole Svanstrom and Batear sources held out; exact cross-source copies move together",
        "counts": {s: sum(r["split"] == s for r in records) for s in ["train", "development", "final"]},
        "source_counts": {s: sum(r["source"] == s for r in records) for s in sorted({r["source"] for r in records})},
        "label_scope": "Publisher-level weak labels only; no fabricated event ground truth",
        "audit": report})
    print(f"Prepared {len(records)} real recordings; manifest and source/day splits saved", flush=True)


if __name__ == "__main__":
    main()
