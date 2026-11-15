#!/usr/bin/env python3
"""Prepare a source-grouped, weak-label 3-second crop study from verified archives.

All eligible original files contribute one predeclared central crop. This is a
clip-classification study, not full-recording event detection. No model scores
are used to select crops, labels, groups or partitions.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from drone_detector.common import digest, file_hash, read_json, write_json, write_jsonl
from drone_detector.data import components, load_manifest
from drone_detector.source_adapters import tut2017_metadata

POLAND = {
    1: ("2023-03-15", "Kielce"),
    2: ("2023-03-15", "Kielce"),
    3: ("2023-04-15", "Gdansk"),
    4: ("2023-04-15", "Gdansk"),
    5: ("2023-04-15", "Gdansk"),
    6: ("2023-04-15", "Gdansk"),
    7: ("2023-04-16", "Debogorze"),
    8: ("2023-04-16", "Debogorze"),
    9: ("2023-04-16", "Debogorze"),
    10: ("2023-04-17", "Debogorze"),
    11: ("2023-04-17", "Debogorze"),
    12: ("2023-04-17", "Debogorze"),
    13: ("2023-04-17", "Debogorze"),
    14: ("2023-04-18", "Lapalice"),
    15: ("2023-04-18", "Lapalice"),
    16: ("2023-04-18", "Lapalice"),
    17: ("2023-04-18", "Lapalice"),
}
PROTOCOL = {
    "schema_version": 1,
    "task": "Weak publisher-label acoustic clip classification with independent source/capture holdouts",
    "crop": "one central3s crop per eligible native file, all samples for shorter files; minimum0.25s; no score-driven selection",
    "canonical_audio": "select physical channel0; float32 decoding/polyphase16kHz; FLOAT WAV; no gain normalization",
    "train": [
        "Poland captures2023-03-15,2023-04-15,2023-04-16",
        "Berne2022-08-03",
        "TUT2017 development",
        "DNC",
    ],
    "development": ["Poland capture2023-04-17", "TUT2017 evaluation"],
    "final": [
        "Poland capture2023-04-18 (new site)",
        "DroneNoise Edzell (one primary microphone/event)",
        "NEAPTIDE (microphone1, processed stress set)",
        "UrbanSound8K",
    ],
    "selection": "fixed linear C=1; fit scalers and heads on TRAIN only; choose one global logit threshold on DEVELOPMENT negatives at1% recording FPR",
    "models": [
        "relative-harmonic spectral summaries + logistic",
        "frozen CLAP + logistic",
        "frozen BEATs + logistic",
    ],
    "weighting": "equal class, then equal source within class, then equal capture/provenance group within source/class, then equal rows within group",
    "metrics": [
        "per-domain positive recall and negative-recording FPR",
        "paired-domain ROC-AUC",
        "macro-source balanced accuracy",
        "duration/level-only control",
    ],
    "limitations": [
        "Labels are inherited from publisher recordings; central crop audibility is not independently annotated.",
        "General-background class labels do not certify continuous drone absence.",
        "Source and class remain partly confounded; domain transfer cannot identify a single causal factor.",
        "No event recall, alarm occupancy, false alarms/hour or detection-delay claim.",
        "Foreign positive sources each have few independent campaigns; clip count is not campaign count.",
    ],
}
URBAN_METADATA_SHA256 = "b207f7a53f9e5eb02a5dd71d58fb8b66c88c24c0637b926dc72d9d400a6a2354"


def make_crop(raw, metadata, cache, container, member):
    audio, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        return None, "empty-or-nonfinite"
    original_duration = len(audio) / rate
    if original_duration < 0.25:
        return None, "shorter-than0.25s"
    samples = min(len(audio), round(3 * rate))
    start = (len(audio) - samples) // 2
    selected = audio[start : start + samples, 0]
    common = gcd(rate, 16000)
    if rate != 16000:
        selected = resample_poly(selected, 16000 // common, rate // common)
    selected = np.asarray(selected, dtype=np.float32)
    if np.max(np.abs(selected)) == 0:
        return None, "exactly-silent-crop"
    decoded_sha = hashlib.sha256(selected.tobytes()).hexdigest()
    identifier = metadata["source"] + "-" + digest([container.name, member])[:24]
    path = cache / "audio" / metadata["source"] / (identifier + ".wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".partial.wav")
        sf.write(temporary, selected, 16000, subtype="FLOAT")
        temporary.replace(path)
    else:
        stored, saved_rate = sf.read(path, dtype="float32")
        if saved_rate != 16000 or not np.array_equal(stored, selected):
            raise ValueError("Changed canonical cache: " + identifier)
    row = {
        "schema_version": 1,
        "id": identifier,
        "path": str(path.resolve()),
        "source_version": "corpus-intake-v3-pinned-originals",
        "origin": "authentic",
        "annotation_status": "weak",
        "events": [],
        "ambiguous": [],
        "duration_seconds": len(selected) / 16000,
        "channel_index": 0,
        "channels": 1,
        "sha256": file_hash(path),
        "decoded_sha256": decoded_sha,
        "parent_id": None,
        "duplicate_review_required": False,
        "provenance_review": "Pinned archive/whole-file checksum, member CRC, publisher original/capture identities; conservative cohort grouping",
        "label_scope": "weak recording label inherited to prospective central crop; not independently confirmed crop audibility",
        "native": {
            "container": container.name,
            "member": member,
            "member_sha256": hashlib.sha256(raw).hexdigest(),
            "sample_rate": rate,
            "channels": audio.shape[1],
            "frames": len(audio),
            "duration_seconds": original_duration,
            "crop_start_frame": start,
            "crop_end_frame": start + samples,
            "selected_channel": 0,
        },
        **metadata,
    }
    return row, None


def tut_lookup(archives, scratch):
    result = {}
    for release in ["development", "evaluation"]:
        folder = archives / ("tut2017-" + release)
        metadata = folder / f"TUT-acoustic-scenes-2017-{release}.meta.zip"
        if not metadata.exists():
            continue
        destination = scratch / "metadata" / release
        for package in folder.glob("*.zip"):
            if not any(t in package.name for t in [".meta.zip", ".doc.zip"]):
                continue
            receipt = read_json(package.with_name(package.name + ".complete.json"))
            if (
                not receipt.get("publisher_checksum_verified")
                or file_hash(package) != receipt["sha256"]
            ):
                raise ValueError("Unverified or changed TUT metadata")
            with zipfile.ZipFile(package) as z:
                for name in z.namelist():
                    if name.endswith("/") or not any(
                        name.endswith(s) for s in ["meta.txt", "map.txt"]
                    ):
                        continue
                    target = destination / Path(name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(name))
        root = destination / f"TUT-acoustic-scenes-2017-{release}"
        records = tut2017_metadata(root, release)
        result[release] = {Path(r["path"]).name: r for r in records}
    return result


def metadata_for(source, name, archive, lookups):
    base = {
        "source": source,
        "label": "drone",
        "split": "train",
        "label_evidence": "Publisher drone measurement description; see pinned source documentation",
    }
    stem = Path(name).stem
    if source == "poland-uav-v1":
        if "/olympusls11/" not in name.lower():
            return None, "non-Olympus or auxiliary measurement"
        drone = int(re.search(r"_D(\d+)_", archive.name, re.I).group(1))
        date, site = POLAND[drone]
        native = re.search(r"_(\d{6})_", stem)
        if native and native.group(1) != date[2:].replace("-", ""):
            return None, "publisher-file-date-conflict"
        base.update(
            {
                "original_id": name,
                "session_id": date + "-" + site,
                "capture_date": date,
                "site": site,
                "drone_id": "D" + str(drone),
                "split": "development"
                if date == "2023-04-17"
                else "final"
                if date == "2023-04-18"
                else "train",
                "speech_marker": "sekwenc" in stem.lower(),
                "device": "Olympus LS11",
            }
        )
    elif source == "idmt-berne-2022":
        flight = re.sub(r"_\d+_\d+$", "", stem)
        base.update(
            {
                "original_id": flight,
                "event_id": stem,
                "session_id": "2022-08-03-Berne",
                "site": "Berne",
                "device": "array physical channel0",
            }
        )
    elif source == "neaptide-v1":
        if not stem.endswith("_1"):
            return None, "simultaneous microphone view; primary channel1 selected"
        base.update(
            {
                "original_id": archive.stem,
                "session_id": "Winterthur-2023-06-campaign",
                "split": "final",
                "event_id": stem.rsplit("_", 1)[0],
                "site": "Winterthur",
                "processed_measurement": True,
                "device": "Behringer ECM8000 microphone1",
            }
        )
    elif source == "dronenoise-edzell-v3":
        if stem.lower().startswith("calib"):
            return None, "calibration"
        if not stem.endswith("_M1"):
            return None, "simultaneous microphone view; primary M1 selected"
        base.update(
            {
                "original_id": stem.rsplit("_M", 1)[0],
                "event_id": stem.rsplit("_M", 1)[0],
                "session_id": "Edzell-2022-08-17",
                "site": "Edzell",
                "split": "final",
                "device": "microphone M1",
            }
        )
    elif source.startswith("tut2017-"):
        release = source.removeprefix("tut2017-")
        item = lookups["tut"][release][Path(name).name]
        base.update(
            {
                "label": "non_drone",
                "split": "train" if release == "development" else "development",
                "original_id": item["original_id"],
                "session_id": item["session_id"],
                "publisher_scene": item["publisher_scene"],
                "publisher_source_interval": item["source_interval_seconds"],
                "label_evidence": item["label_evidence"],
                "device": "TUT binaural channel0",
            }
        )
    elif source == "urbansound8k":
        item = lookups["urban"][Path(name).name]
        base.update(
            {
                "label": "non_drone",
                "split": "final",
                "original_id": "freesound:" + item["fsID"],
                "session_id": "freesound:" + item["fsID"],
                "publisher_class": item["class"],
                "publisher_fold": item["fold"],
                "publisher_source_interval": [float(item["start"]), float(item["end"])],
                "label_evidence": "Publisher foreground sound class "
                + item["class"]
                + "; not exhaustive drone-absence annotation",
            }
        )
    elif source == "dnc":
        base.update(
            {
                "label": "non_drone",
                "split": "train",
                "original_id": name,
                "session_id": None,
                "label_evidence": "Publisher DNC environmental-background collection; not reviewed drone-absent exposure",
            }
        )
    else:
        return None, "unsupported-source"
    return base, None


def process(item, archives, cache, lookups):
    source = item["source_id"]
    path = archives / source / item["filename"]
    receipt = read_json(path.with_name(path.name + ".complete.json"))
    if receipt.get("md5") != item["md5"] or not receipt.get("publisher_checksum_verified"):
        raise ValueError("Source lacks publisher verification: " + str(path))
    if file_hash(path) != receipt["sha256"]:
        raise ValueError("Changed source: " + str(path))
    records = []
    excluded = []

    def audio(name, raw):
        if "__MACOSX" in name or not name.lower().endswith(".wav"):
            return
        metadata, reason = metadata_for(source, name, path, lookups)
        if metadata is None:
            excluded.append({"container": path.name, "member": name, "reason": reason})
            return
        row, reason = make_crop(raw, metadata, cache, path, name)
        if row:
            row["native"]["container_sha256"] = receipt["sha256"]
            records.append(row)
        else:
            excluded.append({"container": path.name, "member": name, "reason": reason})

    if path.suffix.lower() == ".wav":
        audio(path.name, path.read_bytes())
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.filename.lower().endswith(".wav") and "__MACOSX" not in info.filename:
                    # Read every selected WAV member fully so its ZIP CRC is checked.
                    metadata, reason = metadata_for(source, info.filename, path, lookups)
                    if metadata is None:
                        excluded.append(
                            {"container": path.name, "member": info.filename, "reason": reason}
                        )
                        continue
                    audio(info.filename, z.read(info))
    elif path.name.endswith(".tar.gz") or path.suffix == ".tar":
        with tarfile.open(path, mode="r|*") as tar:
            for member in tar:
                if member.isfile() and member.name.lower().endswith(".wav"):
                    stream = tar.extractfile(member)
                    audio(member.name, stream.read())
    return records, excluded


def prepare(archives, cache, urban_csv, wait=False):
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "preparation-complete.json").exists():
        complete = read_json(cache / "preparation-complete.json")
        if complete["manifest_sha256"] != file_hash(cache / "manifest.jsonl"):
            raise ValueError("Finalized study manifest changed")
        print("The accepted study cohort is already finalized; keeping it unchanged", flush=True)
        return
    if file_hash(urban_csv) != URBAN_METADATA_SHA256:
        raise ValueError("UrbanSound metadata differs from the verified release CSV")
    urban_cache = cache / "metadata/urbansound.csv"
    urban_cache.parent.mkdir(parents=True, exist_ok=True)
    if urban_cache.exists() and file_hash(urban_cache) != URBAN_METADATA_SHA256:
        raise ValueError("Changed cached UrbanSound metadata")
    urban_cache.write_bytes(urban_csv.read_bytes())
    protocol_path = cache / "protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != PROTOCOL:
        raise ValueError("Protocol changed")
    write_json(protocol_path, PROTOCOL)
    root = Path(__file__).resolve().parents[1]
    manifest = read_json(root / "data/corpus-intake-v3.json")
    metadata_items = [
        i
        for i in manifest["items"]
        if i["source_id"].startswith("tut2017-")
        and i["filename"].endswith((".meta.zip", ".doc.zip"))
    ]
    while True:
        missing = [
            i["filename"]
            for i in metadata_items
            if not (archives / i["source_id"] / (i["filename"] + ".complete.json")).exists()
        ]
        if not missing:
            break
        if not wait:
            raise ValueError("Complete required TUT metadata acquisition first: " + repr(missing))
        print(json.dumps({"waiting_for_source_metadata": missing}), flush=True)
        time.sleep(10)
    items = [
        i
        for i in manifest["items"]
        if i["filename"].lower().endswith((".wav", ".zip", ".tar.gz"))
        and not any(t in i["filename"] for t in [".meta.zip", ".doc.zip", ".error.zip"])
    ]
    lookups = {
        "tut": tut_lookup(archives, cache),
        "urban": {r["slice_file_name"]: r for r in csv.DictReader(urban_csv.open())},
    }
    pending = list(items)
    while pending:
        moved = False
        for item in list(pending):
            path = archives / item["source_id"] / item["filename"]
            part = cache / "parts" / (digest([item["source_id"], item["filename"]]) + ".json")
            if part.exists():
                pending.remove(item)
                continue
            if not path.with_name(path.name + ".complete.json").exists():
                continue
            records, excluded = process(item, archives, cache, lookups)
            part.parent.mkdir(parents=True, exist_ok=True)
            write_json(
                part,
                {
                    "source": item,
                    "records": records,
                    "excluded": excluded,
                    "protocol_sha256": digest(PROTOCOL),
                },
            )
            pending.remove(item)
            moved = True
            print(
                json.dumps(
                    {
                        "prepared": item["source_id"] + "/" + item["filename"],
                        "records": len(records),
                        "excluded": len(excluded),
                    }
                ),
                flush=True,
            )
        if pending and not wait:
            break
        if pending and not moved:
            time.sleep(10)
    rows = []
    exclusions = []
    for part in sorted((cache / "parts").glob("*.json")):
        data = read_json(part)
        if data["protocol_sha256"] != digest(PROTOCOL):
            raise ValueError("Prepared part protocol changed")
        rows.extend(data["records"])
        exclusions.extend(data["excluded"])
    # Remove duplicate canonical crops without transferring any final example into fitting.
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["decoded_sha256"]].append(row)
    accepted = []
    for group in by_hash.values():
        if len({r["split"] for r in group}) > 1 or len({r["label"] for r in group}) > 1:
            exclusions.extend(
                {"id": r["id"], "reason": "canonical-duplicate-crosses-planned-split-or-class"}
                for r in group
            )
        else:
            selected = min(group, key=lambda r: r["id"])
            accepted.append(selected)
            exclusions.extend(
                {"id": r["id"], "reason": "duplicate-canonical-crop", "retained_id": selected["id"]}
                for r in group
                if r is not selected
            )
    for row, group in zip(accepted, components(accepted), strict=True):
        row["group_id"] = group
    write_jsonl(cache / "manifest.jsonl", accepted)
    if not pending:
        load_manifest(cache / "manifest.jsonl", require_split=True)
    report = {
        "records": len(accepted),
        "counts": {
            str(k): v
            for k, v in Counter((r["source"], r["split"], r["label"]) for r in accepted).items()
        },
        "native_file_hours": sum(r["native"]["duration_seconds"] for r in accepted) / 3600,
        "selected_crop_hours": sum(r["duration_seconds"] for r in accepted) / 3600,
        "provenance_groups": len({r["group_id"] for r in accepted}),
        "exclusions": exclusions,
        "pending": [i["source_id"] + "/" + i["filename"] for i in pending],
        "scope": "Prospective central-crop classification with weak inherited labels; not independently annotated event detection",
    }
    write_json(cache / "inventory.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "exclusions"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urban-csv", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    prepare(args.archives, args.output, args.urban_csv, args.wait)


if __name__ == "__main__":
    main()
