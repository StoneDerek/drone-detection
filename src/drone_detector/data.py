"""Audio inventory, conservative provenance components and pre-window splits."""

import hashlib
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from .audio import decode
from .common import digest, file_hash, finite_number, read_jsonl, write_json, write_jsonl

SPLITS = {"unassigned", "train", "development", "final"}
VERIFIED = {"verified_temporal", "verified_negative"}


def validate_record(row):
    required = ["id", "path", "source", "source_version", "label_evidence"]
    for name in required:
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise ValueError(f"Missing {name}")
    if row.get("schema_version") != 1 or row.get("split") not in SPLITS:
        raise ValueError("Unknown manifest schema/split")
    if row.get("origin") not in {"authentic", "synthetic", "playback", "unknown"}:
        raise ValueError("Unknown audio origin")
    if row.get("label") not in {"drone", "non_drone", "uncertain"}:
        raise ValueError("Invalid class mapping")
    if row.get("annotation_status") not in VERIFIED | {"weak", "unknown"}:
        raise ValueError("Invalid annotation status")
    duration = row.get("duration_seconds")
    if not finite_number(duration) or duration <= 0:
        raise ValueError("Invalid duration")
    for name in ["events", "ambiguous"]:
        last = -1.0
        for interval in row.get(name, []):
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f"Invalid {name} interval")
            a, b = interval
            if not all(finite_number(v) for v in [a, b]) or not (0 <= a < b <= duration):
                raise ValueError(f"Out-of-bounds {name}")
            if a < last:
                raise ValueError(f"Overlapping or unordered {name}")
            last = b
    if row["annotation_status"] == "verified_negative":
        if row["label"] != "non_drone" or row.get("events"):
            raise ValueError("Negative annotation contradicts label/events")
    if row["annotation_status"] == "verified_temporal":
        if row["label"] != "drone" or not row.get("events"):
            raise ValueError("Positive annotation needs temporal events")
    for a, b in row.get("events", []):
        if any(a < d and b > c for c, d in row.get("ambiguous", [])):
            raise ValueError("Event overlaps ambiguous audio; resolve its boundaries")


def components(rows):
    """Union identities transitively; missing provenance collapses a source."""
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def join(i, j):
        parents[find(j)] = find(i)

    seen = {}
    by_id = {r["id"]: i for i, r in enumerate(rows)}
    unknown_sources = {
        (r["source"], r["source_version"]) for r in rows
        if not r.get("session_id") or not r.get("original_id")
    }
    for i, row in enumerate(rows):
        source = (row["source"], row["source_version"])
        keys = []
        if source in unknown_sources:
            keys.append(("unknown-source", *source))
        for field in ["session_id", "original_id"]:
            if row.get(field):
                keys.append((field, *source, row[field]))
        for field in ["sha256", "decoded_sha256", "gain_normalized_sha256"]:
            if row.get(field):
                keys.append((field, row[field]))
        for key in keys:
            if key in seen:
                join(i, seen[key])
            else:
                seen[key] = i
        if row.get("parent_id"):
            if row["parent_id"] not in by_id:
                raise ValueError(f"Parent not present: {row['id']}")
            join(i, by_id[row["parent_id"]])
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row["id"])
    ids = {i: digest(sorted(group))[:20] for i, group in groups.items()}
    return [ids[find(i)] for i in range(len(rows))]


def load_manifest(path, require_split=False):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("Manifest contains no audio; this is not a successful benchmark")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate recording IDs")
    if len({r["path"] for r in rows}) != len(rows):
        raise ValueError("Duplicate paths")
    for row in rows:
        validate_record(row)
    groups = components(rows)
    splits = defaultdict(set)
    for row, group in zip(rows, groups, strict=True):
        if row.get("group_id") != group:
            raise ValueError("Manifest grouping changed; re-audit and re-split")
        splits[group].add(row["split"])
        if require_split and row["split"] == "unassigned":
            raise ValueError("Split the manifest before training/evaluation")
        if require_split and row.get("duplicate_review_required", True):
            raise ValueError("Resolve provenance/excerpt/duplicate review before training/evaluation")
    if any(len(values) != 1 for values in splits.values()):
        raise ValueError("Provenance/duplicate component leaks across splits")
    return rows


def checked_audio(row, manifest_path, sample_rate=16000):
    path = (Path(manifest_path).resolve().parent / row["path"]).resolve()
    if file_hash(path) != row["sha256"]:
        raise ValueError(f"Audio changed since audit: {row['id']}")
    audio, info = decode(path, sample_rate)
    if abs(info["duration_seconds"] - row["duration_seconds"]) > 1 / sample_rate:
        raise ValueError("Duration changed")
    return audio


def audit(root, output, annotations=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Refusing to overwrite an existing audit")
    annotation_rows = read_jsonl(annotations) if annotations else []
    metadata = {r["path"]: r for r in annotation_rows}
    if len(metadata) != len(annotation_rows):
        raise ValueError("Repeated annotation paths")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".ogg"})
    if not files:
        raise ValueError("No audio found")
    discovered = {str(p.relative_to(root)) for p in files}
    if set(metadata) - discovered:
        raise ValueError("Annotations reference missing files")
    rows = []
    for path in files:
        relative = str(path.relative_to(root))
        meta = metadata.get(relative, {})
        audio, info = decode(path)
        content = file_hash(path)
        pcm = np.rint(audio.astype(np.float64) * 32767).clip(-32768, 32767).astype("<i2")
        peak = float(np.max(np.abs(audio)))
        normalized = np.rint(audio / max(peak, 1e-20) * 1024).astype("<i2")
        flags = []
        if info["clipped_fraction"] > 0.001:
            flags.append("clipping")
        if info["rms"] < 1e-5:
            flags.append("silence")
        if info["duration_seconds"] < 3:
            flags.append("short_context")
        row = {
            "schema_version": 1, "id": meta.get("id", digest(relative)[:16]),
            "path": os.path.relpath(path, output.parent),
            "source": meta.get("source", "unknown"),
            "source_version": meta.get("source_version", "unknown"),
            "session_id": meta.get("session_id"), "original_id": meta.get("original_id"),
            "parent_id": meta.get("parent_id"), "origin": meta.get("origin", "unknown"),
            "label": meta.get("label", "uncertain"),
            "annotation_status": meta.get("annotation_status", "unknown"),
            "label_evidence": meta.get("label_evidence", "No annotation supplied"),
            "events": meta.get("events", []), "ambiguous": meta.get("ambiguous", []),
            "device": meta.get("device"), "environment": meta.get("environment"),
            "drone_type": meta.get("drone_type"), "distance_m": meta.get("distance_m"),
            "snr_db": meta.get("snr_db"), "license": meta.get("license"),
            "negative_kind": meta.get("negative_kind"),
            "source_recording": meta.get("source_recording"),
            "provenance_review": meta.get("provenance_review"),
            "duplicate_review_required": not bool(meta.get("provenance_review")),
            "sha256": content,
            "decoded_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
            "gain_normalized_sha256": hashlib.sha256(normalized.tobytes()).hexdigest(),
            "split": "unassigned", "quality_flags": flags, **info,
        }
        validate_record(row)
        rows.append(row)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate recording IDs in annotations")
    for row, group in zip(rows, components(rows), strict=True):
        row["group_id"] = group
    write_jsonl(output, rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row["id"])
    report = {
        "schema_version": 1, "manifest_sha256": file_hash(output), "recordings": len(rows),
        "total_hours": sum(r["duration_seconds"] for r in rows) / 3600,
        "groups": dict(groups), "flags": {r["id"]: r["quality_flags"] for r in rows if r["quality_flags"]},
        "unresolved_provenance": [r["id"] for r in rows if r["duplicate_review_required"]],
        "duplicate_method_limit": "Exact/gain-normalized whole-file hashes cannot rule out lossy copies or shifted/overlapping excerpts. Human provenance review remains mandatory.",
        "known_authentic_recordings": sum(r["origin"] == "authentic" for r in rows),
    }
    write_json(output.with_suffix(".audit.json"), report)
    return report


def split_manifest(manifest, output, seed=42, final_sources=()):
    rows = load_manifest(manifest)
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite an existing split")
    if any(r["split"] != "unassigned" for r in rows):
        raise ValueError("Manifest is already split; create a new version explicitly")
    groups = sorted({r["group_id"] for r in rows})
    held = {r["group_id"] for r in rows if r["source"] in final_sources}
    if set(final_sources) - {r["source"] for r in rows}:
        raise ValueError("Unknown final source")
    remaining = sorted(set(groups) - held)
    if len(remaining) < (2 if held else 3):
        raise ValueError("Too few independent provenance groups for three splits")
    rng = np.random.default_rng(seed)
    chosen = None
    # Label-aware group allocation is part of prespecified dataset construction,
    # not model selection. Never separate recordings within a component.
    for _ in range(512):
        shuffled = list(rng.permutation(remaining))
        n = max(1, round(len(shuffled) * 0.2))
        assignments = {g: "train" for g in groups}
        if held:
            assignments.update({g: "final" for g in held})
            assignments.update({g: "development" for g in shuffled[:n]})
        else:
            assignments.update({g: "final" for g in shuffled[:n]})
            assignments.update({g: "development" for g in shuffled[n:2*n]})
        labels = defaultdict(set)
        for row in rows:
            if row["annotation_status"] in VERIFIED:
                labels[assignments[row["group_id"]]].add(row["label"])
        if all(labels[split] == {"drone", "non_drone"} for split in ["train", "development", "final"]):
            chosen = assignments
            break
    if chosen is None:
        raise ValueError("Cannot put both verified classes in all splits without leakage; obtain independent groups")
    for row in rows:
        row["split"] = chosen[row["group_id"]]
        full_path = (Path(manifest).resolve().parent / row["path"]).resolve()
        row["path"] = os.path.relpath(full_path, output.resolve().parent)
    write_jsonl(output, rows)
    load_manifest(output, require_split=True)
    definition = {
        "schema_version": 1, "seed": seed, "manifest_sha256": file_hash(output),
        "parent_manifest_sha256": file_hash(manifest), "final_sources": list(final_sources),
        "groups": {s: sorted(g for g, value in chosen.items() if value == s)
                   for s in ["train", "development", "final"]},
        "counts": {s: sum(r["split"] == s for r in rows) for s in ["train", "development", "final"]},
        "method": "Seeded group shuffle, accept first allocation with both verified labels in every split; 60/20/20 rounded by groups; reserved final sources take precedence.",
    }
    write_json(output.with_suffix(".splits.json"), definition)
    return definition
