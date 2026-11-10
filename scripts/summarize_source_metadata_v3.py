"""Summarize downloaded source metadata; never score audio or invent labels.

Inputs and retrieval identities are documented in reports/source-scope-audit-v3.
Durations are metadata interval unions, not independently reviewed field hours.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


def union_seconds(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    end = 0.0
    for start, stop in sorted(intervals):
        if not (math.isfinite(start) and math.isfinite(stop) and 0 <= start < stop):
            raise ValueError(f"Invalid source interval: {start}, {stop}")
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def summarize(root: Path) -> dict:
    esc = csv_rows(root / "esc50.csv")
    urban = csv_rows(root / "urbansound.csv")
    index = json.loads((root / "urbansound-index.json").read_text())
    expected = index["metadata"]["UrbanSound8K.csv"][1]
    actual = hashlib.md5((root / "urbansound.csv").read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("UrbanSound CSV differs from the soundata release index")

    intervals = defaultdict(list)
    urban_folds = defaultdict(set)
    for row in urban:
        intervals[row["fsID"]].append((float(row["start"]), float(row["end"])))
        urban_folds[row["fsID"]].add(row["fold"])
    esc_folds = defaultdict(set)
    for row in esc:
        esc_folds[row["src_file"]].add(row["fold"])
    shared = sorted(set(esc_folds) & set(intervals), key=int)

    tut = {}
    tut_groups = {}
    for release in ("development", "evaluation"):
        directory = root / "tut-docs" / f"TUT-acoustic-scenes-2017-{release}"
        mapping_file = directory / "evaluation_setup" / "map.txt"
        mapping = {}
        if mapping_file.exists():
            for line in mapping_file.read_text().splitlines():
                original, renamed = line.split("\t")
                if renamed in mapping:
                    raise ValueError(f"Duplicate mapped filename: {renamed}")
                mapping[renamed] = original
        groups = defaultdict(list)
        classes = Counter()
        files = set()
        for line in (directory / "meta.txt").read_text().splitlines():
            name, scene, group = line.split("\t")
            if name in files:
                raise ValueError(f"Duplicate metadata filename: {name}")
            files.add(name)
            original = mapping.get(name, name)
            parent, start, stop = Path(original).stem.rsplit("_", 2)
            if parent != group:
                raise ValueError(f"Source mapping disagrees with metadata: {name}")
            groups[group].append((float(start), float(stop)))
            classes[scene] += 1
        tut_groups[release] = set(groups)
        tut[release] = {
            "clips": len(files),
            "original_recording_ids": len(groups),
            "class_clip_counts": dict(sorted(classes.items())),
            "summed_interval_seconds": sum(b - a for v in groups.values() for a, b in v),
            "represented_interval_union_seconds": sum(union_seconds(v) for v in groups.values()),
            "interval_overlaps_within_parent": any(
                abs(sum(b - a for a, b in v) - union_seconds(v)) > 1e-6
                for v in groups.values()
            ),
            "independent_capture_sessions_verified": False,
        }

    tree = json.loads((root / "alemadi-tree.json").read_text())
    if tree.get("truncated"):
        raise ValueError("Incomplete Al-Emadi repository tree")
    wavs = [m for m in tree["tree"] if m["type"] == "blob" and m["path"].endswith(".wav")]
    folder_counts = Counter(str(Path(m["path"]).parent) for m in wavs)
    binary = {m["sha"] for m in wavs if m["path"].startswith("Binary_Drone_Audio/")}
    multi = {m["sha"] for m in wavs if m["path"].startswith("Multiclass_Drone_Audio/")}
    esc_names = {row["filename"] for row in esc}
    names = [Path(m["path"]).name for m in wavs if "/unknown/" in m["path"]]
    esc_derivatives = [name for name in names if Path(name).stem[:-1] + ".wav" in esc_names]

    return {
        "kind": "metadata-only measurements; not accepted training or evaluation exposure",
        "urbansound8k": {
            "clips": len(urban),
            "freesound_parent_ids": len(intervals),
            "csv_matches_soundata_index_md5": actual,
            "summed_clip_interval_seconds": sum(b - a for v in intervals.values() for a, b in v),
            "represented_interval_union_seconds": sum(union_seconds(v) for v in intervals.values()),
            "parent_ids_crossing_publisher_folds": sum(len(v) > 1 for v in urban_folds.values()),
        },
        "esc50": {
            "clips": len(esc),
            "publisher_seconds_per_clip": 5,
            "nominal_file_seconds": len(esc) * 5,
            "freesound_parent_ids": len(esc_folds),
            "parent_ids_crossing_publisher_folds": sum(len(v) > 1 for v in esc_folds.values()),
            "original_time_offsets_available_in_this_csv": False,
            "fresh_blind_test_status": "previously consumed development source",
        },
        "cross_source": {
            "esc50_urbansound_shared_freesound_ids": shared,
            "esc50_urbansound_shared_parent_count": len(shared),
            "shared_ids_prove_same_parent_not_identical_excerpt": True,
            "tut_development_evaluation_shared_parent_ids": sorted(
                tut_groups["development"] & tut_groups["evaluation"]
            ),
        },
        "tut2017": tut,
        "alemadi": {
            "git_tree_sha": tree["sha"],
            "tree_truncated": tree["truncated"],
            "wav_path_count": len(wavs),
            "unique_wav_git_blob_ids": len({m["sha"] for m in wavs}),
            "folder_wav_counts": dict(sorted(folder_counts.items())),
            "binary_unique_wav_blobs": len(binary),
            "multiclass_unique_wav_blobs": len(multi),
            "binary_multiclass_shared_wav_blobs": len(binary & multi),
            "unknown_paths_matching_esc50_name_plus_chunk_digit": len(esc_derivatives),
            "matching_esc50_original_filenames": len(
                {Path(name).stem[:-1] + ".wav" for name in esc_derivatives}
            ),
            "filename_mapping_is_not_payload_verification": True,
            "unique_original_capture_seconds": None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(args.metadata_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
