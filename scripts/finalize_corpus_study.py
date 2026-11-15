#!/usr/bin/env python3
"""Freeze data quality decisions before any model is fitted or holdout is scored."""

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from drone_detector.common import digest, file_hash, read_json, write_json, write_jsonl
from drone_detector.data import components, load_manifest

POLICY = {
    "version": 2,
    "selection": "No model predictions, embeddings or listening judgments enter these decisions",
    "publisher_errors": "Exclude a fixed crop if it intersects a publisher-marked faulty interval",
    "duplicates": "Exact native-member/canonical checks plus quantized gain-normalized crop candidates verified by waveform residual",
    "cross_split_duplicate": "Quarantine every member of a duplicate component spanning splits or labels",
    "within_split_duplicate": "Keep lexicographically smallest identifier",
    "historical_exposure": "Quarantine UrbanSound8K parents shared with previously consumed ESC-50 material, using the published 94-parent metadata intersection",
    "limits": "This does not rule out shifted, resampled or lossy cross-source excerpts, or overlap in model pretraining",
}
URBAN_METADATA_SHA256 = "b207f7a53f9e5eb02a5dd71d58fb8b66c88c24c0637b926dc72d9d400a6a2354"


def intersects_fault(row, intervals):
    native = row["native"]
    start = native["crop_start_frame"] / native["sample_rate"]
    end = native["crop_end_frame"] / native["sample_rate"]
    return any(start < b and end > a for a, b in intervals)


def publisher_errors(archives):
    result = defaultdict(list)
    for path in sorted(archives.glob("tut2017-*/*.error.zip")):
        receipt = read_json(path.with_name(path.name + ".complete.json"))
        if not receipt.get("publisher_checksum_verified") or file_hash(path) != receipt["sha256"]:
            raise ValueError("Unverified publisher error annotations")
        with zipfile.ZipFile(path) as package:
            for name in package.namelist():
                if not name.endswith("error.txt"):
                    continue
                for line in package.read(name).decode("utf-8-sig").splitlines():
                    member, a, b, *_ = line.split("\t")
                    result[(path.parent.name, Path(member).name)].append((float(a), float(b)))
    return result


def deduplicate(rows):
    """Verify gain candidates; recording copies stay in the same component."""
    parents = list(range(len(rows)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    candidates = defaultdict(list)
    verified_gain_pairs = []
    for index, row in enumerate(rows):
        for kind, value in [
            ("canonical", row["decoded_sha256"]),
            ("native", row["native"]["member_sha256"]),
        ]:
            key = (kind, value)
            if key in seen:
                parents[root(index)] = root(seen[key])
            else:
                seen[key] = index
        audio, rate = sf.read(row["path"], dtype="float32")
        if rate != 16000 or file_hash(row["path"]) != row["sha256"]:
            raise ValueError("Canonical input changed")
        peak = float(np.max(np.abs(audio)))
        normalized = np.rint(audio.astype(np.float64) / peak * 4096).astype("<i2")
        key = hashlib.sha256(normalized.tobytes()).hexdigest()
        for other in candidates[key]:
            reference, _ = sf.read(rows[other]["path"], dtype="float64")
            signal = audio.astype(np.float64)
            if len(reference) != len(signal):
                continue
            gain = float(np.dot(reference, signal) / np.dot(reference, reference))
            residual = float(np.linalg.norm(signal - gain * reference) / np.linalg.norm(signal))
            if residual <= 1e-5:
                parents[root(index)] = root(other)
                verified_gain_pairs.append(
                    {
                        "ids": [rows[other]["id"], row["id"]],
                        "gain": gain,
                        "relative_residual": residual,
                    }
                )
        candidates[key].append(index)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[root(index)].append(row)
    accepted = []
    exclusions = []
    for group in groups.values():
        conflict = len({r["split"] for r in group}) > 1 or len({r["label"] for r in group}) > 1
        if conflict:
            exclusions.extend(
                {"id": r["id"], "reason": "verified-duplicate-crosses-split-or-label"}
                for r in group
            )
        else:
            retained = min(group, key=lambda r: r["id"])
            accepted.append(retained)
            exclusions.extend(
                {
                    "id": r["id"],
                    "reason": "verified-duplicate-within-split",
                    "retained_id": retained["id"],
                }
                for r in group
                if r is not retained
            )
    return accepted, exclusions, verified_gain_pairs


def finalize(prepared, archives):
    marker = prepared / "preparation-complete.json"
    if marker.exists():
        receipt = read_json(marker)
        if receipt["manifest_sha256"] != file_hash(prepared / "manifest.jsonl"):
            raise ValueError("Finalized manifest changed")
        if receipt["policy_sha256"] == digest(POLICY):
            return receipt
        if (
            receipt["policy_sha256"]
            != "403b0624673d5dbf035986108e9043d443e5c0d625bcb79e894a6f9318e5ac1e"
            or (prepared / "final-evaluation-authorized.json").exists()
        ):
            raise ValueError("Quality policy changed; preserve the old study")
        for name in ["preparation-complete", "quality-finalization", "quality-policy"]:
            (prepared / (name + "-v1.json")).write_bytes((prepared / (name + ".json")).read_bytes())
        (prepared / "manifest-before-history-review.jsonl").write_bytes(
            (prepared / "manifest.jsonl").read_bytes()
        )
        marker.unlink()
    if (prepared / "final-evaluation-authorized.json").exists():
        raise ValueError("Cannot revise a cohort after models have been frozen")
    inventory = read_json(prepared / "inventory.json")
    if inventory["pending"]:
        raise ValueError("Source preparation is not complete")
    urban_metadata = prepared / "metadata/urbansound.csv"
    if file_hash(urban_metadata) != URBAN_METADATA_SHA256:
        raise ValueError("UrbanSound metadata differs from the verified release CSV")
    intake = read_json(Path(__file__).resolve().parents[1] / "data/corpus-intake-v3.json")
    distribution = []
    for item in intake["items"]:
        path = archives / item["source_id"] / item["filename"]
        receipt = read_json(path.with_name(path.name + ".complete.json"))
        if (
            path.stat().st_size != item["bytes"]
            or receipt.get("md5") != item["md5"]
            or not receipt.get("publisher_checksum_verified")
        ):
            raise ValueError("Missing or inconsistent source acquisition")
        distribution.append(
            {
                "source": item["source_id"],
                "filename": item["filename"],
                "bytes": item["bytes"],
                "md5": receipt["md5"],
                "sha256": receipt["sha256"],
            }
        )
    write_json(prepared / "verified-distributions.json", distribution)
    write_json(prepared / "quality-policy.json", POLICY)
    original = prepared / "manifest-before-quality.jsonl"
    if not original.exists():
        original.write_bytes((prepared / "manifest.jsonl").read_bytes())
    rows = load_manifest(original, require_split=True)
    before_sha = file_hash(original)
    errors = publisher_errors(archives)
    history_path = (
        Path(__file__).resolve().parents[1] / "reports/source-scope-audit-v3/metadata-summary.json"
    )
    history = read_json(history_path)["cross_source"]
    known_parents = set(history["esc50_urbansound_shared_freesound_ids"])
    if len(known_parents) != history["esc50_urbansound_shared_parent_count"]:
        raise ValueError("Historical parent metadata is inconsistent")
    clean = []
    excluded = []
    for row in rows:
        if row["source"] == "urbansound8k" and row["original_id"].split(":")[-1] in known_parents:
            if row["split"] != "final":
                raise ValueError("Unexpected historical-parent assignment")
            excluded.append(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "original_id": row["original_id"],
                    "reason": "historically-consumed-ESC50-parent",
                }
            )
            continue
        intervals = errors.get((row["source"], Path(row["native"]["member"]).name), [])
        if intersects_fault(row, intervals):
            excluded.append(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "reason": "crop-intersects-publisher-error",
                    "intervals_seconds": intervals,
                }
            )
        else:
            clean.append(row)
    accepted, duplicates, gain_pairs = deduplicate(clean)
    excluded.extend(duplicates)
    for row, group in zip(accepted, components(accepted), strict=True):
        row["group_id"] = group
    candidate = prepared / "manifest-quality-candidate.jsonl"
    write_jsonl(candidate, accepted)
    load_manifest(candidate, require_split=True)
    for split in ["train", "development"]:
        if {r["label"] for r in accepted if r["split"] == split} != {"drone", "non_drone"}:
            raise ValueError("Both classes required in training and development")
    tut_development = {r["original_id"] for r in accepted if r["source"] == "tut2017-development"}
    tut_evaluation = {r["original_id"] for r in accepted if r["source"] == "tut2017-evaluation"}
    if tut_development & tut_evaluation:
        raise ValueError("TUT original identities overlap across releases")
    candidate.replace(prepared / "manifest.jsonl")
    report = {
        "input_manifest_sha256": before_sha,
        "manifest_sha256": file_hash(prepared / "manifest.jsonl"),
        "policy_sha256": digest(POLICY),
        "implementation_sha256": file_hash(__file__),
        "verified_distribution_files": len(distribution),
        "verified_distribution_bytes": sum(r["bytes"] for r in distribution),
        "urban_metadata_sha256": URBAN_METADATA_SHA256,
        "historical_overlap_reference_sha256": file_hash(history_path),
        "historically_shared_parent_ids": sorted(known_parents),
        "records": len(accepted),
        "counts": dict(Counter(f"{r['source']}/{r['split']}/{r['label']}" for r in accepted)),
        "capture_group_counts": {
            split: len({r["group_id"] for r in accepted if r["split"] == split})
            for split in ["train", "development", "final"]
        },
        "native_file_hours": sum(r["native"]["duration_seconds"] for r in accepted) / 3600,
        "selected_crop_hours": sum(r["duration_seconds"] for r in accepted) / 3600,
        "additional_exclusions": excluded,
        "verified_gain_duplicate_pairs": gain_pairs,
        "tut_original_ids_disjoint": True,
        "pretraining_and_lossy_excerpt_overlap": "Not ruled out",
    }
    write_json(prepared / "quality-finalization.json", report)
    write_json(
        marker,
        {
            k: v
            for k, v in report.items()
            if k not in {"additional_exclusions", "verified_gain_duplicate_pairs"}
        },
    )
    print(json.dumps(read_json(marker), indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.prepared, args.archives)


if __name__ == "__main__":
    main()
