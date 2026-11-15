"""Original-source metadata adapters; labels remain auxiliary and splits explicit.

Metadata catalogs can be created before audio arrives. Preparing a manifest reads
native headers and hashes whole files, without modifying or resampling originals.
Scene/class labels never become verified drone-absent time or audible intervals.
"""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import soundfile as sf

from .common import digest, file_hash, finite_number, read_jsonl, write_json, write_jsonl
from .data import SPLITS, components, validate_record


def _relative(value):
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe source-relative path: {value}")
    return path.as_posix()


def _interval(start, end):
    start, end = float(start), float(end)
    if not all(finite_number(v) for v in (start, end)) or not 0 <= start < end:
        raise ValueError("Invalid original-recording interval")
    return [start, end]


def _base(source, version, path, evidence):
    return {
        "id": f"{source}-{digest([version, path])[:20]}",
        "path": _relative(path), "source": source, "source_version": version,
        "origin": "authentic", "label": "non_drone", "annotation_status": "weak",
        "label_evidence": evidence,
        "label_scope": "Auxiliary publisher class/scene label; drone absence unverified",
        "events": [], "ambiguous": [], "split": "unassigned",
        "duplicate_review_required": True,
    }


def tut2017_metadata(root, release):
    """Parse a release directory containing meta.txt and its optional rename map."""
    if release not in {"development", "evaluation"}:
        raise ValueError("Unknown TUT2017 release")
    root = Path(root)
    metadata_path = root / "meta.txt"
    mapping_path = root / "evaluation_setup" / "map.txt"
    metadata_hash = file_hash(metadata_path)
    mapping_hash = file_hash(mapping_path) if mapping_path.exists() else None
    mapping = {}
    if mapping_path.exists():
        for line in mapping_path.read_text().splitlines():
            if not line.strip():
                continue
            original, renamed = line.split("\t")
            if renamed in mapping:
                raise ValueError("Duplicate TUT renamed path")
            mapping[renamed] = original
    rows, seen = [], set()
    for line in metadata_path.read_text().splitlines():
        if not line.strip():
            continue
        path, scene, original_id = line.split("\t")
        if path in seen:
            raise ValueError("Duplicate TUT metadata path")
        seen.add(path)
        try:
            parent, start, end = PurePosixPath(mapping.get(path, path)).stem.rsplit("_", 2)
        except ValueError as error:
            raise ValueError("TUT renamed excerpts require evaluation_setup/map.txt") from error
        if parent != original_id:
            raise ValueError("TUT metadata and original-recording mapping disagree")
        row = _base("tut2017", release, path,
                    f"Publisher TUT2017 {release} acoustic-scene label: {scene}; "
                    "not an audited drone-absence annotation")
        row.update({
            "original_id": original_id, "session_id": f"recording:{original_id}",
            "session_id_scope": "Original recording; capture-day/site independence unresolved",
            "source_recording": mapping.get(path, path),
            "source_interval_seconds": _interval(start, end),
            "publisher_scene": scene, "publisher_release": release,
            "metadata_sha256": metadata_hash, "mapping_sha256": mapping_hash,
            "provenance_keys": [f"tut2017-recording:{original_id}"],
            "license": "Publisher TUT research EULA; preserve original terms",
        })
        rows.append(row)
    if not rows:
        raise ValueError("Empty TUT metadata")
    return rows


def urbansound8k_metadata(csv_path):
    """Keep Freesound parent IDs and overlapping excerpt offsets across folds."""
    csv_path = Path(csv_path)
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        metadata = list(csv.DictReader(stream))
    rows, seen = [], set()
    metadata_hash = file_hash(csv_path)
    for item in metadata:
        name, parent, fold = item["slice_file_name"], item["fsID"], item["fold"]
        if not parent.isdigit() or not fold.isdigit() or not 1 <= int(fold) <= 10:
            raise ValueError("Invalid UrbanSound parent/fold")
        if name in seen or PurePosixPath(name).name != name:
            raise ValueError("Duplicate or invalid UrbanSound filename")
        seen.add(name)
        if name.split("-", 1)[0] != parent:
            raise ValueError("UrbanSound filename disagrees with Freesound parent")
        path = f"audio/fold{fold}/{name}"
        row = _base("urbansound8k", "1.0.0", path,
                    f"Publisher UrbanSound8K event class: {item['class']}; "
                    "not an audited drone-absence annotation")
        row.update({
            "original_id": f"freesound:{parent}", "session_id": f"freesound:{parent}",
            "session_id_scope": "Freesound parent recording; independent capture session unresolved",
            "source_recording": f"https://freesound.org/s/{parent}/",
            "source_interval_seconds": _interval(item["start"], item["end"]),
            "publisher_class": item["class"], "publisher_class_id": item["classID"],
            "publisher_fold": int(fold), "publisher_salience": item["salience"],
            "metadata_sha256": metadata_hash, "provenance_keys": [f"freesound:{parent}"],
            "license": "CC BY-NC 3.0; preserve per-source credits",
        })
        rows.append(row)
    if not rows:
        raise ValueError("Empty UrbanSound metadata")
    return rows


def dnc_metadata(root):
    """Inventory DNC audio without inventing per-file session independence."""
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".wav", ".flac", ".ogg"}:
            continue
        relative = path.relative_to(root).as_posix()
        row = _base("dnc", "depositonce-11645", relative,
                    "Publisher DNC environmental-background collection; "
                    "drone absence and event timing not independently reviewed")
        row.update({
            "original_id": relative, "session_id": None,
            "session_id_scope": "Unknown; conservatively group the entire source",
            "publisher_category": next((part for part in path.relative_to(root).parts[:-1]
                                        if part in {"mechanic", "melodic", "quiet"}), None),
            "provenance_keys": [], "license": "MIT as linked by original audio record",
        })
        rows.append(row)
    if not rows:
        raise ValueError("No DNC audio found")
    return rows


def generic_metadata(path, source, source_version):
    """Import explicit per-file metadata; no labels inferred from drone/speech names.

    Suitable for supplying reviewed Berne/Polish/NEAPTIDE mappings later. Each row
    needs path, label and label_evidence. Missing annotation strength stays weak.
    """
    rows = []
    metadata_hash = file_hash(path)
    for item in read_jsonl(path):
        if item.get("label") not in {"drone", "non_drone", "uncertain"}:
            raise ValueError("Generic metadata requires an explicit valid label")
        if not isinstance(item.get("label_evidence"), str) or not item["label_evidence"].strip():
            raise ValueError("Generic metadata requires explicit label evidence")
        row = _base(source, source_version, item["path"], item["label_evidence"])
        row.update(item)
        row.update({"path": _relative(item["path"]), "source": source,
                    "source_version": source_version, "split": "unassigned",
                    "duplicate_review_required": True, "metadata_sha256": metadata_hash})
        row["label_scope"] = item.get("label_scope", "Explicit per-file annotation; "
                                     "audibility and timing supported only as documented")
        if "source_interval_seconds" in row:
            row["source_interval_seconds"] = _interval(*row["source_interval_seconds"])
        rows.append(row)
    if not rows:
        raise ValueError("Empty generic metadata")
    return rows


def metadata_plan(plan, plan_directory):
    """Resolve explicit source roots; this function does not read audio samples."""
    base = Path(plan_directory).resolve()
    records = []
    for source in plan["sources"]:
        def resolve(name):
            return (base / source[name]).resolve()
        adapter = source["adapter"]
        if adapter == "tut2017":
            rows = tut2017_metadata(resolve("metadata_root"), source["release"])
        elif adapter == "urbansound8k":
            rows = urbansound8k_metadata(resolve("metadata"))
        elif adapter == "dnc":
            rows = dnc_metadata(resolve("audio_root"))
        elif adapter == "generic":
            rows = generic_metadata(resolve("metadata"), source["source"], source["source_version"])
        else:
            raise ValueError(f"Unsupported source adapter: {adapter}")
        for row in rows:
            if "channel_index" in source:
                if "channel_index" in row and row["channel_index"] != source["channel_index"]:
                    raise ValueError("Source channel choice conflicts with per-file mapping")
                row["channel_index"] = source["channel_index"]
            records.append({**row, "audio_root": str(resolve("audio_root"))})
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("Empty source plan or repeated recording IDs")
    return records


def prepare_manifest(records, output, review=None):
    """Hash native audio, link provenance, and emit an unassigned schema1 manifest.

    Review is an explicit mapping id -> {split, provenance_review}; no automatic
    random split is made. Cross-source keys link to actual parent rows so existing
    data.components/load_manifest enforce the same conservative components.
    """
    output = Path(output).resolve()
    if output.exists() or output.with_suffix(".intake.json").exists():
        raise ValueError("Preserve existing manifest/intake report")
    review = {} if review is None else review
    rows = []
    for item in records:
        root = Path(item["audio_root"]).resolve()
        path = (root / _relative(item["path"])).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Audio path escapes its source root")
        facts = sf.info(path)
        if facts.frames <= 0 or facts.samplerate <= 0:
            raise ValueError(f"Empty or invalid native audio: {item['id']}")
        if facts.channels > 2 and "channel_index" not in item:
            raise ValueError("Choose one physical array channel explicitly before manifest intake")
        channel = item.get("channel_index", 0)
        if (not isinstance(channel, int) or isinstance(channel, bool)
                or not 0 <= channel < facts.channels):
            raise ValueError("Physical channel index is invalid for native audio")
        duration = facts.frames / facts.samplerate
        interval = item.get("source_interval_seconds")
        if interval and abs(duration - (interval[1] - interval[0])) > 1 / facts.samplerate + 1e-5:
            raise ValueError(f"Header duration disagrees with source offsets: {item['id']}")
        row = {key: value for key, value in item.items() if key != "audio_root"}
        row.update({
            "schema_version": 1, "path": os.path.relpath(path, output.parent),
            "sha256": file_hash(path), "duration_seconds": duration,
            "original_sample_rate": facts.samplerate, "channels": facts.channels,
            "native_frames": facts.frames, "native_subtype": facts.subtype,
            "channel_index": channel, "channel_policy": f"select_physical_channel_{channel}",
            "original_bytes_unchanged": True, "split": "unassigned",
            "duplicate_review_required": True,
        })
        if row["id"] in review:
            decision = review[row["id"]]
            if decision.get("split") not in SPLITS - {"unassigned"}:
                raise ValueError("Reviewed assignments require an explicit train/development/final split")
            note = decision.get("provenance_review")
            if not isinstance(note, str) or not note.strip():
                raise ValueError("Reviewed assignments require provenance evidence")
            row.update({"split": decision["split"], "provenance_review": note,
                        "duplicate_review_required": False})
        validate_record(row)
        rows.append(row)
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Empty intake or duplicate recording IDs")
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("An audio file occurs more than once in this intake")
    if set(review) - {row["id"] for row in rows}:
        raise ValueError("Review references unknown recording IDs")
    # Build connected parent links. A row may join several provenance keys; union
    # the identities first so no existing parent link gets overwritten/lost.
    by_id = {row["id"]: i for i, row in enumerate(rows)}
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        if row.get("parent_id"):
            if row["parent_id"] not in by_id:
                raise ValueError("Parent row is missing from the intake")
            parents[find(i)] = find(by_id[row["parent_id"]])
        for key in row.get("provenance_keys", []):
            if not isinstance(key, str) or not key:
                raise ValueError("Provenance keys must be nonempty strings")
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    for i, row in enumerate(rows):
        row.pop("parent_id", None)
        if find(i) != i:
            row["parent_id"] = rows[find(i)]["id"]
    split_groups = defaultdict(set)
    for row, group in zip(rows, components(rows), strict=True):
        row["group_id"] = group
        split_groups[group].add(row["split"])
    if any(len(values) != 1 for values in split_groups.values()):
        raise ValueError("Related recordings cross assignment splits; review whole components together")
    write_jsonl(output, rows)
    report = {
        "schema_version": 1, "manifest_sha256": file_hash(output),
        "recordings": len(rows), "provenance_components": len(split_groups),
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "annotation_counts": dict(Counter(row["annotation_status"] for row in rows)),
        "unresolved_provenance": [row["id"] for row in rows if row["duplicate_review_required"]],
        "summed_file_seconds": sum(row["duration_seconds"] for row in rows),
        "scope": "Native headers and whole-file hashes only. File seconds are not independent "
                 "capture time. Weak class labels do not certify drone-absent hours or audible events.",
    }
    write_json(output.with_suffix(".intake.json"), report)
    return report
