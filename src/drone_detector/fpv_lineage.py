"""Conservative filename lineage for the complete public FPV archive.

Names establish declared relationships, never waveform identity. Missing
originals and publisher copy suffixes remain explicit unresolved outcomes.
"""

import collections
import hashlib
import re
from pathlib import PurePosixPath

import numpy as np

SPLIT = re.compile(r"^(.*)_(train|validate|test)(?:(?:_segment)?_(\d+)_(\d+))?$")
COPY = re.compile(r"\(\d+\)$")


def identifier(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def original_intervals(entries):
    return [e for e in entries if "/labeled_data/" in e["path"]
            and "/ProcessedAudioSplits/" not in e["path"]
            and e["path"].lower().endswith(".wav")]


def declared_lineage(entries, annotations):
    originals = original_intervals(entries)
    aliases = collections.defaultdict(list)
    devices = collections.defaultdict(set)
    for e in originals:
        p = PurePosixPath(e["path"])
        alias = "_".join([p.parts[-2], p.parts[-3], p.stem])
        aliases[alias].append(e["path"])
        devices[p.parts[-3]].add(str(p.parent.parent))
    for annotation in annotations:
        if annotation["archive"] != "Fpv.zip":
            continue
        p = PurePosixPath(annotation["member"]).parent
        devices[p.name].add(str(p))
    device_keys = sorted(devices, key=lambda value: (-len(value), value))
    rows = []
    for e in entries:
        p = PurePosixPath(e["path"])
        if "/ProcessedAudioSplits/" not in e["path"] or p.suffix.lower() != ".wav":
            continue
        stem = p.stem
        copy_match = COPY.search(stem)
        copy_suffix = copy_match is not None
        stripped = COPY.sub("", stem)
        match = SPLIT.fullmatch(stripped)
        if match is None:
            rows.append(dict(member=e["path"], status="unparsed", candidate_parents=[],
                             candidate_views=[], candidate_capture_groups=[]))
            continue
        alias, partition, start, end = match.groups()
        cls, _, remainder = alias.partition("_")
        expected_partition = "train" if p.parts[4] == "augmented_train" else p.parts[4]
        consistent = partition == expected_partition and cls == p.parts[3]
        parents = aliases.get(alias, [])
        keys = [key for key in device_keys if remainder.startswith(key + "_")]
        longest = len(keys[0]) if keys else 0
        views = sorted({view for key in keys if len(key) == longest for view in devices[key]})
        if parents:
            views = sorted({str(PurePosixPath(parent).parent.parent) for parent in parents})
        captures = sorted({"/".join(PurePosixPath(view).parts[:3]) for view in views})
        status = "declared_parent_present" if len(parents) == 1 else "ambiguous_parent" if parents else "original_parent_absent"
        if copy_suffix:
            status = "unresolved_copy_suffix"
        if not consistent:
            status = "inconsistent_class_or_partition"
        start_ms, end_ms = (int(start), int(end)) if start is not None else (None, None)
        if start_ms is not None and end_ms <= start_ms:
            status = "invalid_declared_offsets"
        rows.append(dict(
            member=e["path"], member_id=identifier(e["path"]), source_alias=alias,
            partition=partition, export_kind=p.parts[4], class_label=cls,
            start_ms=start_ms, end_ms=end_ms, copy_suffix=copy_suffix, status=status,
            copy_tag=copy_match.group(0) if copy_match else "",
            candidate_parents=list(parents), candidate_views=views,
            candidate_capture_groups=captures,
        ))
    return rows


def union_length(intervals):
    total, left, right = 0, None, None
    for start, end in sorted(intervals):
        if left is None:
            left, right = start, end
        elif start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + (right - left if left is not None else 0)


def summarize(rows):
    parent_partitions = collections.defaultdict(set)
    captures = collections.defaultdict(set)
    augmented = collections.defaultdict(list)
    for r in rows:
        if "source_alias" not in r:
            continue
        parent_partitions[r["source_alias"]].add(r["partition"])
        for group in r["candidate_capture_groups"]:
            captures[identifier(group)].add(r["partition"])
        if r["export_kind"] == "augmented_train" and r["start_ms"] is not None:
            augmented[r["source_alias"]].append((r["start_ms"], r["end_ms"]))
    intervals = [pair for group in augmented.values() for pair in group]
    summed = sum(b - a for a, b in intervals) / 1000
    union = sum(union_length(group) for group in augmented.values()) / 1000
    return dict(
        exported_wav_members=len(rows), status_counts=dict(collections.Counter(r["status"] for r in rows)),
        uniquely_declared_capture_members=sum(len(r["candidate_capture_groups"]) == 1 for r in rows),
        declared_source_aliases=len(parent_partitions),
        declared_source_aliases_in_multiple_partitions=sum(len(v) > 1 for v in parent_partitions.values()),
        candidate_capture_groups=len(captures),
        candidate_capture_groups_in_multiple_partitions=sum(len(v) > 1 for v in captures.values()),
        augmented_training_windows=len(intervals),
        augmented_declared_summed_seconds=summed,
        augmented_declared_timeline_union_seconds=union,
        augmented_declared_overlap_fraction=1 - union / summed if summed else None,
        offset_windows_shorter_than_250ms=sum(
            r.get("start_ms") is not None and r["end_ms"] - r["start_ms"] < 250 for r in rows
        ),
        duration_scope="Filename-declared intervals per export alias only; neither verified waveform offsets nor unique physical capture time",
        partition_scope="Shared declared parent/capture across publisher partitions; does not itself prove overlapping samples",
    )


def decoded_hash(audio, rate):
    audio = np.asarray(audio, dtype=np.float32)
    h = hashlib.sha256(f"{rate}:{audio.shape}".encode())
    h.update(audio.tobytes())
    return h.hexdigest()


def exact_subsequence(source, fragment):
    """Return a unique native-sample offset; never guess through transforms."""
    if source.ndim != 2 or fragment.ndim != 2 or source.shape[1] != fragment.shape[1]:
        return dict(status="incompatible_channels")
    return ExactNativeIndex(source).match(fragment)


class ExactNativeIndex:
    """Reuse one source byte string while checking all its declared exports."""

    def __init__(self, source):
        if source.ndim != 2:
            raise ValueError("Expected native frames by channels")
        self.shape, self.dtype = source.shape, source.dtype
        self.payload = source.tobytes()

    def match(self, fragment):
        if fragment.ndim != 2 or self.shape[1] != fragment.shape[1]:
            return dict(status="incompatible_channels")
        if self.dtype != fragment.dtype:
            return dict(status="incompatible_sample_dtype")
        if not len(fragment) or len(fragment) > self.shape[0]:
            return dict(status="invalid_fragment_length")
        needle = fragment.tobytes()
        width = self.shape[1] * self.dtype.itemsize
        offsets, start = [], 0
        while len(offsets) < 2:
            index = self.payload.find(needle, start)
            if index < 0:
                break
            if index % width == 0:
                offsets.append(index // width)
            start = index + 1
        if not offsets:
            return dict(status="no_exact_match")
        if len(offsets) > 1:
            return dict(status="ambiguous_repeated_waveform")
        return dict(status="unique_exact_match", start_frame=offsets[0], frames=len(fragment))
