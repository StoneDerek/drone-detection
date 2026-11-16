#!/usr/bin/env python3
"""Audit every FPV export and original interval; never run a detector."""

import argparse
import collections
import hashlib
import io
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from drone_detector.common import file_hash, read_json, write_json, write_jsonl
from drone_detector.corpus_intake import verify
from drone_detector.fpv_lineage import (
    declared_lineage, decoded_hash, exact_subsequence, original_intervals, summarize,
)


def read_audio(archive, member):
    payload = archive.read(member)  # A full member read checks its ZIP CRC.
    if payload[4:8] == b"ftyp" and payload[8:12] == b"M4A ":
        # Four supplied .wav names actually contain M4A/AAC. Decode at native
        # rate/channels to PCM16, the representation of the publisher's WAV
        # exports. No rate conversion, gain fitting or stereo averaging.
        with tempfile.NamedTemporaryFile(suffix=".m4a") as source:
            source.write(payload)
            source.flush()
            info = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
                "-f", "mov", "-select_streams", "a:0", "-show_entries",
                "stream=codec_name,sample_rate,channels", "-of", "json", source.name,
            ], timeout=30))["streams"][0]
            if info["codec_name"] != "aac":
                raise ValueError("Unreviewed codec in a WAV-named M4A member")
            rate, channels = int(info["sample_rate"]), int(info["channels"])
            decoded = subprocess.check_output([
                "ffmpeg", "-v", "error", "-nostdin", "-protocol_whitelist", "file,pipe",
                "-threads", "1", "-f", "mov", "-i", source.name, "-map", "0:a:0",
                "-acodec", "pcm_s16le", "-f", "s16le", "pipe:1",
            ], timeout=60)
        audio = np.frombuffer(decoded, dtype="<i2").reshape(-1, channels).astype(np.float32) / 32768
        subtype, container, decoder = "AAC_to_PCM16", "M4A", "ffmpeg-native-rate-channels-pcm16"
    else:
        info = sf.info(io.BytesIO(payload))
        audio, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
        subtype, container, decoder = info.subtype, info.format, "libsndfile-float32"
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Empty or nonfinite decoded member")
    return audio, rate, dict(
        sha256=hashlib.sha256(payload).hexdigest(),
        decoded_float32_sha256=decoded_hash(audio, rate), frames=len(audio),
        channels=audio.shape[1], sample_rate=rate, subtype=subtype,
        container=container, decoder=decoder,
        duration_seconds=len(audio) / rate, digital_zero=not np.any(audio),
    )


def partition_key(row):
    return row["source_alias"], row["partition"], row["copy_tag"]


def waveforms(archive, rows, originals, work):
    details, source_edges, slice_edges, failures = {}, [], [], []
    bases = collections.defaultdict(list)
    chunks = collections.defaultdict(list)
    for row in rows:
        if "source_alias" not in row:
            continue
        (bases if row["start_ms"] is None else chunks)[partition_key(row)].append(row)
    assigned = collections.defaultdict(list)
    for key, group in bases.items():
        parents = group[0]["candidate_parents"]
        if len(group) == 1 and len(parents) == 1:
            assigned[parents[0]].append((key, group[0]))
    processed = set()

    def load(member):
        try:
            audio, rate, meta = read_audio(archive, member)
            details[member] = meta
            return audio, rate
        except (ValueError, RuntimeError) as error:
            failures.append(dict(member=member, error=str(error)))
            return None, None

    def process_base(key, row, source=None, source_rate=None):
        member = row["member"]
        audio, rate = load(member)
        processed.add(member)
        if audio is None:
            return
        edge = dict(member=member, candidate_parents=row["candidate_parents"])
        if source is None:
            edge["status"] = "original_waveform_not_uniquely_available"
        elif source_rate != rate:
            edge["status"] = "different_sample_rates"
        else:
            edge.update(exact_subsequence(source, audio))
        source_edges.append(edge)
        for child in chunks.get(key, []):
            child_audio, child_rate = load(child["member"])
            processed.add(child["member"])
            outcome = dict(member=child["member"], partition_parent=member)
            if child_audio is None:
                outcome["status"] = "decode_failed"
            elif child_rate != rate or child_audio.shape[1] != audio.shape[1]:
                outcome["status"] = "incompatible_rate_or_channels"
            else:
                start = child["start_ms"] * rate // 1000
                end = child["end_ms"] * rate // 1000
                expected = audio[start:min(end, len(audio))]
                outcome.update(start_frame=start, declared_end_frame=end)
                if np.array_equal(expected, child_audio):
                    outcome["status"] = "exact_declared_slice"
                elif (len(expected) < len(child_audio) <= len(expected) + 2 * (rate // 1000 + 1)
                      and np.array_equal(child_audio[:len(expected)], expected)
                      and not np.any(child_audio[len(expected):])):
                    outcome["status"] = "exact_slice_with_short_zero_tail"
                    outcome["zero_tail_frames"] = len(child_audio) - len(expected)
                else:
                    outcome["status"] = "not_an_exact_declared_slice"
            slice_edges.append(outcome)

    for count, original in enumerate(originals, 1):
        audio, rate = load(original["path"])
        for key, row in assigned.get(original["path"], []):
            process_base(key, row, audio, rate)
        if count % 25 == 0:
            print("Verified original interval members", count, "/", len(originals), flush=True)
    for key, group in bases.items():
        if len(group) == 1 and group[0]["member"] not in processed:
            process_base(key, group[0])
    for count, row in enumerate(rows, 1):
        member = row["member"]
        if member not in processed:
            load(member)
            processed.add(member)
            slice_edges.append(dict(member=member, status="unique_partition_parent_unavailable"))
        if count % 2000 == 0:
            print("Accounted export members", count, "/", len(rows), flush=True)
    identities = collections.defaultdict(list)
    by_member = {r["member"]: r for r in rows}
    for member, info in details.items():
        if member in by_member:
            identities[info["decoded_float32_sha256"]].append(by_member[member])
    write_jsonl(work / "decoded-members.jsonl", [dict(member=k, **v) for k, v in details.items()])
    write_jsonl(work / "partition-to-original.jsonl", source_edges)
    write_jsonl(work / "export-to-partition.jsonl", slice_edges)
    write_json(work / "decode-failures.json", failures)
    return dict(
        original_members_requested=len(originals), export_members_requested=len(rows),
        total_members_decoded=len(details), decode_failures=len(failures),
        native_format_counts=dict(collections.Counter(
            f"{v['sample_rate']}Hz/{v['channels']}ch/{v['subtype']}" for v in details.values()
        )),
        partition_to_original=dict(collections.Counter(e["status"] for e in source_edges)),
        export_to_partition=dict(collections.Counter(e["status"] for e in slice_edges)),
        decoded_identity_duplicate_groups=sum(len(v) > 1 for v in identities.values()),
        decoded_identity_groups_across_partitions=sum(len({r['partition'] for r in v}) > 1 for v in identities.values()),
        decoded_identity_groups_across_labels=sum(len({r['class_label'] for r in v}) > 1 for v in identities.values()),
        exports_shorter_than_250ms=sum(details[r["member"]]["duration_seconds"] < .25 for r in rows if r["member"] in details),
        decoded_identity_scope="WAV native-rate/channel float32 samples; misnamed M4A files use fixed native-rate/channel FFmpeg PCM16 decoding. Raw SHA-256 is retained separately. No resampling or gain fitting.",
        permanent_waveform_files_written=0, detector_scores_generated=0,
        ffmpeg_version=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("work/positive-source-discovery-v1"))
    parser.add_argument("--work", type=Path, default=Path("work/fpv-lineage-v1"))
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    catalogue = args.metadata / "Fpv.zip.directory.json"
    entries = read_json(catalogue)
    rows = declared_lineage(entries, read_json(args.metadata / "annotation-manifest.json"))
    result = dict(metadata=summarize(rows), directory_sha256=file_hash(catalogue), scope="Source qualification, not detector evaluation")
    write_jsonl(args.work / "lineage.jsonl", rows)
    write_json(args.work / "summary.json", result)
    if args.archive:
        observed = verify(args.archive, dict(bytes=2152999558, md5="b06064de415a1d8fdf16b3e508ba96a0"))
        result["archive"] = dict(bytes=args.archive.stat().st_size, **observed, publisher_md5_verified=True)
        with zipfile.ZipFile(args.archive) as archive:
            actual = [(i.filename, i.file_size, i.compress_size, i.CRC, i.header_offset) for i in archive.infolist() if not i.is_dir()]
            expected = [(e['path'], e['bytes'], e['compressed_bytes'], e['crc32'], e['header_offset']) for e in entries]
            if actual != expected:
                raise ValueError("Downloaded archive differs from the previously reviewed directory")
            result["waveforms"] = waveforms(archive, rows, original_intervals(entries), args.work)
        write_json(args.work / "summary.json", result)
        write_json(args.work / "waveform-audit-complete.json", dict(summary_sha256=file_hash(args.work / "summary.json")))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
