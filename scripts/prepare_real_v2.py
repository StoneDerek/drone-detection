#!/usr/bin/env python3
"""Broader training and a genuinely fresh, versioned real-audio assessment.

The previous final is explicitly consumed development evidence. Fresh DDL flight
days and the April 28 Batear session remain unscored until the new freeze.
"""

import collections
import argparse
import concurrent.futures
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

from ddl_audio import RangeReader
from fetch_public_audio import fetch_blob, get
from drone_detector.common import file_hash, read_json, read_jsonl, write_json, write_jsonl
from drone_detector.data import audit, components, load_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Verify and reuse this script\'s completed downloads')
    args = parser.parse_args()
    root = Path("runs/real-v2")
    output = root / "corpus"
    if output.exists() and not args.resume:
        raise ValueError("Preserve existing corpus; choose a new version")
    if (output / 'manifest-v2.jsonl').exists():
        raise ValueError('The corpus is already complete; preserve its frozen identity')
    (output / "audio").mkdir(parents=True, exist_ok=True)
    annotations = []
    original_annotations = {r["id"]: r for r in read_jsonl("runs/real-v1/corpus/annotations.jsonl")}
    for row in load_manifest("runs/real-v1/corpus/manifest-v1.jsonl", True):
        # The unknown-parent SNR background excerpt could share a flight with
        # other D3Set files. Exclude it rather than asserting independence.
        if row.get("negative_kind") == "same-array-background":
            continue
        meta = dict(original_annotations[row["id"]])
        if row["source"] == "svanstrom":
            split = "train"  # Entire source; do not split uncertain originals.
        elif row["split"] == "final":
            split = "development"  # Previously assessed and now explicitly reused.
        else:
            split = row["split"]
        meta["split_choice"] = split
        source_path = Path("runs/real-v1/corpus") / row["path"]
        destination = output / meta['path']
        if destination.exists():
            if file_hash(destination) != file_hash(source_path):
                raise ValueError('Existing corpus file differs from the verified source')
        else:
            shutil.copyfile(source_path, destination)
        annotations.append(meta)
    # Fresh negatives from the 40 ESC categories never used in v1. Evaluation
    # only under ESC-50's CC BY-NC terms, with raw audio kept out of the repo.
    public = Path("runs/real-v1/public")
    import csv
    esc = list(csv.DictReader((public / "metadata/esc50.csv").open()))
    known_ids = {r["src_file"] for r in esc if r["esc10"].lower() == "true"}
    fresh = [r for r in esc if r["esc10"].lower() != "true" and r["src_file"] not in known_ids]
    tree = read_json(public / "metadata/esc10-tree.json")
    blobs = {r["path"]: r for r in tree["tree"]}
    version = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"

    def esc_negative(row):
        path = "audio/" + row["filename"]
        identity = "esc50-fresh-" + Path(row["filename"]).stem
        task = {"path": f"audio/{identity}.wav", "git_blob": blobs[path]["sha"],
                "url": f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{version}/{path}"}
        result = fetch_blob(task, output)
        return {"id": identity, "path": task["path"], "source": "esc50-fresh-negatives",
            "source_version": version, "session_id": "freesound-" + row["src_file"],
            "original_id": "freesound-" + row["src_file"], "origin": "authentic", "label": "non_drone",
            "annotation_status": "weak", "events": [], "ambiguous": [],
            "label_evidence": "Published environmental-sound category, no independent acoustic absence verification",
            "negative_kind": row["category"], "license": "CC-BY-NC-3.0, assessment only, not used to fit model",
            "split_choice": "final", "provenance_review": "Official Git content verified. No original Freesound ID shared with prior ESC-10 corpus. All derivatives grouped by source ID.",
            "source_recording": {"url": task["url"], "sha256": result["sha256"]}}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for index, meta in enumerate(pool.map(esc_negative, fresh), 1):
            annotations.append(meta)
            if index % 200 == 0:
                print(f"Verified fresh negative recordings {index}/{len(fresh)}", flush=True)
    # DDL provides field labels, flight/session IDs and short sequence fragments.
    # Reconstruct three contiguous 10 s excerpts per flight, selected by sequence
    # position (10%, 50%, 90%), never by a model score. Never concatenate gaps.
    inventory = read_json(root / "ddl-inventory.json")
    flights = collections.defaultdict(list)
    for entry in inventory["entries"]:
        path = Path(entry["path"])
        stem = path.stem
        if path.suffix.lower() != ".wav" or stem[31:32] != "R" or stem[14:18] not in {"MINI", "PRO4", "XXXX"}:
            raise ValueError("Unexpected real DDL filename; inspect before labeling")
        seq = int(stem.rsplit("-", 1)[-1])
        flights[path.parent.name].append({**entry, "sequence": seq,
                                         "class": stem[14:18], "reported_range_m": int(stem[21:24])})
    reader = RangeReader(cache_dir=root / 'ddl-blocks')
    excerpts, sample_log = [], []
    block_indices = set()
    for entries in flights.values():
        ordered = sorted(entries, key=lambda r: r['sequence'])
        nominal_size = collections.Counter(r['size'] for r in ordered if r['size'] > 44).most_common(1)[0][0]
        possible = [i for i in range(len(ordered) - 99)
                    if all(ordered[j]['size'] == nominal_size for j in range(i, i + 100))
                    and all(ordered[j + 1]['sequence'] == ordered[j]['sequence'] + 1 for j in range(i, i + 99))]
        if not possible:
            continue
        for fraction in [.1, .5, .9]:
            desired = round((len(ordered) - 100) * fraction)
            start_index = min(possible, key=lambda i: abs(i - desired))
            for entry in ordered[start_index:start_index + 100]:
                start = entry['header_offset']
                end = start + entry['compressed_size'] + len(entry['path'].encode()) + 4096
                block_indices.update(range(start // reader.block_size, end // reader.block_size + 1))

    def prefetch(index):
        disk = reader.cache_dir / f'{index}.bin'
        checksum = disk.with_suffix('.sha256')
        start = index * reader.block_size
        end = min(reader.length - 1, start + reader.block_size - 1)
        if disk.exists() and checksum.exists():
            if disk.stat().st_size != end - start + 1 or file_hash(disk) != checksum.read_text().strip():
                raise ValueError('Cached ZIP block content mismatch')
            return
        req = urllib.request.Request(reader.url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(req, timeout=120) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{reader.length}':
                raise ValueError('Range prefetch response mismatch')
            payload = response.read()
        if len(payload) != end - start + 1:
            raise ValueError('Incomplete compressed data block')
        disk.write_bytes(payload)
        checksum.write_text(hashlib.sha256(payload).hexdigest() + '\n')

    print(f'Prefetching {len(block_indices)} selected real-audio ZIP blocks with six workers', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for count, _ in enumerate(pool.map(prefetch, sorted(block_indices)), 1):
            if count % 50 == 0:
                print(f'Verified ZIP blocks {count}/{len(block_indices)}', flush=True)
    with zipfile.ZipFile(reader) as archive:
        issues = []
        for number, (flight, entries) in enumerate(sorted(flights.items()), 1):
            ordered = sorted(entries, key=lambda r: r["sequence"])
            nominal_size = collections.Counter(r['size'] for r in ordered if r['size'] > 44).most_common(1)[0][0]
            issues.extend({'path': r['path'], 'bytes': r['size'],
                           'reason': 'empty WAV' if r['size'] <= 44 else 'nonstandard fragment length'}
                          for r in ordered if r['size'] != nominal_size)
            probe_entry = next(r for r in ordered if r['size'] == nominal_size)
            probe = sf.info(io.BytesIO(archive.read(probe_entry['path'])))
            fragments = round(10 / probe.duration)
            if fragments < 1 or abs(fragments * probe.duration - 10) > 1 / probe.samplerate:
                raise ValueError('DDL fragments do not form an exact ten-second context')
            possible = [i for i in range(len(ordered) - fragments + 1)
                        if all(ordered[j]['size'] == nominal_size for j in range(i, i + fragments))
                        and all(ordered[j + 1]['sequence'] == ordered[j]['sequence'] + 1
                               for j in range(i, i + fragments - 1))]
            if not possible:
                issues.append({'path': flight, 'reason': 'no ten-second contiguous nominal-length run'})
                print(f'Excluded unusable flight {flight} by file-quality criteria', flush=True)
                continue
            for fraction in [.1, .5, .9]:
                desired = round((len(ordered) - fragments) * fraction)
                if not possible:
                    raise ValueError("No contiguous ten-second excerpt for flight")
                start = min(possible, key=lambda i: abs(i - desired))
                chosen = ordered[start:start + fragments]
                if len({r["class"] for r in chosen}) != 1:
                    raise ValueError("Mixed labels require temporal annotation")
                chunks, rate, channels = [], None, None
                for entry in chosen:
                    payload = archive.read(entry["path"])  # ZIP verifies member CRC.
                    audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
                    if rate is not None and (rate != sample_rate or channels != audio.shape[1]):
                        raise ValueError("Inconsistent recording format")
                    rate, channels = sample_rate, audio.shape[1]
                    if abs(len(audio) / rate - probe.duration) > 1 / rate:
                        raise ValueError("Unexpected DDL fragment duration")
                    chunks.append(audio[:, 0])  # One physical microphone, not array averaging.
                    sample_log.append({"path": entry["path"], "sha256": hashlib.sha256(payload).hexdigest(),
                                       "crc32": entry["crc32"], "bytes": len(payload)})
                label = "non_drone" if chosen[0]["class"] == "XXXX" else "drone"
                identity = f"ddl-{flight}-{round(fraction * 100):02d}"
                relative = "audio/" + identity + ".wav"
                buffer = io.BytesIO()
                sf.write(buffer, np.concatenate(chunks), rate, format='WAV', subtype='PCM_16')
                content = buffer.getvalue()
                destination = output / relative
                if destination.exists() and file_hash(destination) != hashlib.sha256(content).hexdigest():
                    raise ValueError('Existing DDL excerpt differs from verified reconstruction')
                if not destination.exists():
                    destination.write_bytes(content)
                day = flight.split("-")[0]
                split = "train" if day == "290321" else "final"
                meta = {"id": identity, "path": relative, "source": "ddl-real",
                    "source_version": "zenodo-6459183-real", "session_id": "capture-day-" + day,
                    "original_id": flight, "origin": "authentic", "label": label,
                    "annotation_status": "weak", "events": [], "ambiguous": [],
                    "label_evidence": "DDL publisher real-field class metadata; audible drone boundaries not independently annotated",
                    "license": "CC-BY-4.0", "device": "DDL array physical microphone 0",
                    "drone_type": chosen[0]["class"], "environment": "Published real field capture",
                    "split_choice": split, "provenance_review": "Real archive only; CRC-checked original segments. Contiguous sequence reconstruction. Flight excerpts kept together and entire capture days separated. No synthesized DDL audio fetched.",
                    "source_recording": {"url": inventory["source"], "flight_id": flight,
                        "sequences": [r["sequence"] for r in chosen], "selected_channel": 0,
                        "original_channels": channels, "original_sample_rate": rate,
                        "fragment_seconds": probe.duration,
                        "reported_range_min_m": min(r["reported_range_m"] for r in chosen),
                        "reported_range_max_m": max(r["reported_range_m"] for r in chosen),
                        "whole_archive_md5_verified": False}}
                annotations.append(meta)
                excerpts.append(meta)
            print(f"Reconstructed real DDL flight {number}/{len(flights)}", flush=True)
    write_json(root / 'ddl-quality-exclusions.json', {'criteria': 'Metadata/file quality only, before prediction; contiguous nominal-length fragments required',
                                                   'excluded_members': issues})
    write_jsonl(root / "ddl-selected-members.jsonl", sample_log)
    write_json(root / "ddl-fetch.json", {"source": inventory["source"],
        "metadata": "Zenodo DOI 10.5281/zenodo.6459183", "bytes_transferred": reader.transferred,
        "member_crc_checks": len(sample_log), "member_sha256_recorded": True,
        "publisher_whole_md5": inventory["publisher_archive_md5"], "whole_archive_md5_verified": False,
        "reason": "Partial HTTPS range fetch; every selected ZIP member CRC and its content SHA are retained. No assertion of full-archive MD5 verification.",
        "synthetic_archive_downloaded": False})
    # A second new field session, with both publisher-labeled ambient and DJI
    # recordings. Telemetry range is descriptive, not verified acoustic audibility.
    batear_version = "d0d6e67d9e7d167be9e0db249ef6a44acb52f732"
    base = f"https://raw.githubusercontent.com/batear-io/batear-datasets/{batear_version}/"
    tree = json.loads(get(f"https://api.github.com/repos/batear-io/batear-datasets/git/trees/{batear_version}?recursive=1"))
    candidates = [r for r in tree["tree"] if "/20260428/" in r["path"] and r["path"].lower().endswith(".wav")]

    def field(row):
        path = row["path"]
        pointer = get(base + urllib.parse.quote(path))
        expected = re.search(rb"oid sha256:([0-9a-f]{64})", pointer).group(1).decode()
        identity = "batear-new-" + Path(path).stem
        relative = "audio/" + identity + ".wav"
        destination = output / relative
        data = destination.read_bytes() if destination.exists() else get(f"https://media.githubusercontent.com/media/batear-io/batear-datasets/{batear_version}/" + urllib.parse.quote(path))
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Field LFS hash mismatch")
        if not destination.exists():
            destination.write_bytes(data)
        return {"id": identity, "path": relative, "source": "batear-new-day", "source_version": batear_version,
            "session_id": "2026-04-28", "original_id": identity, "origin": "authentic",
            "label": "non_drone" if "/ambient/" in path else "drone", "annotation_status": "weak",
            "events": [], "ambiguous": [], "label_evidence": "Publisher field directory and flight metadata, acoustic audibility not independently verified",
            "license": "MIT", "split_choice": "final", "device": "Publisher field recorder; see original metadata",
            "drone_type": path.split("/")[2] if "/DJI/" in path else None,
            "negative_kind": "same-session-rural" if "/ambient/" in path else None,
            "provenance_review": "New April 28 capture day, not used in v1 assessment or v2 development. Every original LFS SHA-256 verified; whole day grouped.",
            "source_recording": {"url": base + urllib.parse.quote(path), "sha256": expected}}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        annotations.extend(pool.map(field, candidates))
    write_jsonl(output / "annotations.jsonl", annotations)
    audit(output, output / "inventory.jsonl", output / "annotations.jsonl")
    rows = load_manifest(output / "inventory.jsonl")
    choices = {r["id"]: r["split_choice"] for r in annotations}
    for row in rows:
        row["split"] = choices[row["id"]]
    priority = {"train": 0, "development": 1, "final": 2}
    for group in set(components(rows)):
        members = [r for r in rows if r["group_id"] == group]
        split = max((r["split"] for r in members), key=priority.get)
        if any(r['source'] == 'svanstrom' for r in members) and split != 'final':
            split = 'train'
        for row in members:
            row["split"] = split
    write_jsonl(output / "manifest-v2.jsonl", rows)
    load_manifest(output / "manifest-v2.jsonl", True)
    write_json(root / "data-plan.json", {"manifest_sha256": file_hash(output / "manifest-v2.jsonl"),
        "previous_final_is_development": True, "previous_Svanstrom_source_now_training": True,
        "dataset_revision": "v2.1",
        "unknown_parent_SNR_background_excluded": True,
        "fresh_assessment": "Previously unused DDL capture day (31 March), previously unused Batear capture day (28 April), and previously unused ESC source IDs/classes",
        "DDL_selection": "Three contiguous 10 s excerpts at 10/50/90% per flight, metadata-only selection before any model scoring. Entire capture days grouped. Excerpts are not independent trials.",
        "counts": {s: collections.Counter(r["label"] for r in rows if r["split"] == s)
                   for s in ["train", "development", "final"]},
        "groups": {s: len({r["group_id"] for r in rows if r["split"] == s})
                   for s in ["train", "development", "final"]},
        "field_claim": "Not established; publisher weak labels and only a few positive capture-day groups"})
    print(f"Prepared version 2: {len(rows)} real recordings with a fresh assessment split", flush=True)


if __name__ == "__main__":
    main()
