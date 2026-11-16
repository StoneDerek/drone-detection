#!/usr/bin/env python3
"""Summarize complete cached source catalogues without decoding/scoring audio."""

import argparse
import collections
import csv
import hashlib
import json
import math
import re
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def identity(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(raw):
    original = raw
    match = re.fullmatch(r"(background|drone|uav|electric_uav|eletric_uav)(?:_?\d+)?", raw.strip().lower())
    if match is None:
        raise ValueError("Unresolved publisher label: " + original)
    raw = match.group(1)
    if raw == "background":
        return "negative"
    if raw in {"drone", "uav", "electric_uav", "eletric_uav"}:
        return "positive"
    raise ValueError("Unresolved publisher label: " + raw)


def annotations(root):
    documents = read(root / "annotation-manifest.json")
    rows = []
    for item in documents:
        path = root / "field-annotations" / item["local_name"]
        if identity(path) != item["sha256"]:
            raise ValueError("Publisher annotation bytes changed")
        intervals = []
        raw_labels = collections.Counter()
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            fields = line.split(maxsplit=2)
            if len(fields) != 3:
                raise ValueError("Unsupported annotation row; do not silently skip it")
            start, end = map(float, fields[:2])
            if not math.isfinite(start + end) or start < 0 or end <= start:
                raise ValueError("Invalid annotation interval")
            raw_labels[fields[2]] += 1
            intervals.append((start, end, label(fields[2])))
        conflicts = []
        for index, (start, end, cls) in enumerate(intervals):
            for a, b, other in intervals[:index]:
                overlap = min(end, b) - max(start, a)
                if overlap > 0 and cls != other:
                    conflicts.append(overlap)
        rows.append(dict(
            archive=item["archive"], member_sha256=item["sha256"],
            intervals=len(intervals), raw_label_counts=dict(raw_labels),
            opposite_label_overlap_pairs=len(conflicts),
            opposite_label_overlap_pairs_over_10ms=sum(x > 0.01 for x in conflicts),
            maximum_opposite_label_overlap_seconds=max(conflicts, default=0),
        ))
    return dict(
        documents=len(rows), parsed_intervals=sum(x["intervals"] for x in rows),
        documents_with_opposite_labels_overlapping=sum(x["opposite_label_overlap_pairs"] > 0 for x in rows),
        documents_with_opposite_labels_overlapping_over_10ms=sum(x["opposite_label_overlap_pairs_over_10ms"] > 0 for x in rows),
        maximum_opposite_label_overlap_seconds=max(x["maximum_opposite_label_overlap_seconds"] for x in rows),
        per_document=rows,
        scope="Text integrity only. Annotation endpoints are not verified waveform offsets or independently reviewed audibility. Parallel phone views are not summed as elapsed capture time.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.input
    summary = read(root / "summary.json")
    output = dict(
        scope="Publisher metadata, complete ZIP central directories and every TXT annotation; no audio members extracted, waveforms decoded or detector scores generated",
        source_records=summary["records"],
        field_release=dict(archives=[], annotations=annotations(root)),
        input_sha256={},
    )
    for archive in ["Fpv.zip", "Electric Uavs.zip", "Uav.zip"]:
        path = root / (archive + ".directory.json")
        entries = read(path)
        wavs = [e for e in entries if e["path"].lower().endswith(".wav")]
        split = [e for e in wavs if "/ProcessedAudioSplits/" in e["path"]]
        labeled = [e for e in wavs if "/labeled_data/" in e["path"] and "/ProcessedAudioSplits/" not in e["path"]]
        crc = collections.Counter((e["bytes"], e["crc32"]) for e in labeled + split)
        keys = {Path(e["path"]).parts[2] for e in labeled}
        output["input_sha256"][path.name] = identity(path)
        source = next(x for x in summary["field_archives"] if x["archive"] == archive)
        output["field_release"]["archives"].append(dict(
            archive=archive, publisher_bytes=source["publisher_bytes"],
            publisher_md5=source["publisher_md5"], whole_archive_checksum_verified=False,
            wav_members=len(wavs), original_labeled_interval_wav_members=len(labeled),
            publisher_split_export_wav_members=len(split),
            other_wav_members=len(wavs) - len(labeled) - len(split),
            candidate_labeled_folder_keys=len(keys),
            size_crc_duplicate_candidates=sum(count > 1 for count in crc.values()),
            size_crc_is_not_decoded_identity_proof=True,
        ))
    with (root / "cio-catalog.csv").open() as stream:
        takes = list(csv.DictReader(stream))
    output["cio_indoor"] = dict(
        takes=len(takes), raw_reference_seconds=sum(float(x["raw_reference_s"]) for x in takes),
        publisher_sidecar_date_counts=dict(collections.Counter(x["sidecar_timestamp"][:10] for x in takes)),
        independent_capture_sessions_verified=False, primary_audio_candidate="raw_reference.wav; native mono reference",
        role="Candidate auxiliary indoor training; not a ground-field confirmation set",
    )
    for name in ["cio-catalog.csv", "cio-README.md", "cio-channel_audit.csv", "annotation-manifest.json"]:
        output["input_sha256"][name] = identity(root / name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
