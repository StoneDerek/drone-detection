#!/usr/bin/env python3
"""Reserve new public challenges and repurpose the consumed UrbanSound8K test.

Selection uses publisher metadata and deterministic crop rules, never scores.
The preparation plan is saved before fitting. Whole FSD volumes are checked;
WSU and AuDroK use explicitly bounded source verification, recorded as such.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from drone_detector.common import digest, file_hash, read_json, read_jsonl, write_json, write_jsonl
from drone_detector.corpus_intake import verify
from drone_detector.data import components, load_manifest
from drone_detector.split_zip import SplitZip

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ddl_audio import RangeReader

ROOT = Path(__file__).resolve().parents[1]
METADATA = {
    "FSD50K.ground_truth.zip": "ca27382c195e37d2269c4c866dd73485",
    "FSD50K.metadata.zip": "b9ea0c829a411c1d42adb9da539ed237",
    "FSD50K.doc.zip": "3516162b82dc2945d3e7feba0904e800",
}
ESC_SHA = "ca660da60191a97de289983a05821c9382d852a38a2ba8428980816b68cf6246"
URBAN_SHA = "b207f7a53f9e5eb02a5dd71d58fb8b66c88c24c0637b926dc72d9d400a6a2354"
MECHANICAL = {
    "Drill",
    "Power_tool",
    "Mechanical_fan",
    "Air_conditioning",
    "Jackhammer",
    "Vehicle_horn_and_car_horn_and_honking",
    "Air_horn_and_truck_horn",
    "Train_horn",
    "Chainsaw",
    "Lawn_mower",
    "Sawing",
    "Hair_dryer",
    "Electric_shaver_and_electric_razor",
    "Motorcycle",
    "Car",
    "Truck",
    "Bus",
    "Motor_vehicle_(road)",
}


def checked_metadata(work):
    for filename, expected in METADATA.items():
        path = work / filename
        if hashlib.md5(path.read_bytes()).hexdigest() != expected:
            raise ValueError("FSD publisher metadata checksum mismatch")
    if file_hash(work / "esc50.csv") != ESC_SHA:
        raise ValueError("ESC-50 metadata changed")


def reserve_fsd(work, urban_metadata):
    checked_metadata(work)
    if file_hash(urban_metadata) != URBAN_SHA:
        raise ValueError("Urban metadata changed")
    known = {r["src_file"] for r in csv.DictReader((work / "esc50.csv").open())}
    known |= {r["fsID"] for r in csv.DictReader(urban_metadata.open())}
    with zipfile.ZipFile(work / "FSD50K.metadata.zip") as z:
        info = json.loads(z.read("FSD50K.metadata/eval_clips_info_FSD50K.json"))
        all_info = info | json.loads(z.read("FSD50K.metadata/dev_clips_info_FSD50K.json"))
        linked = {v["uploader"] for k, v in all_info.items() if k in known}
        rows = list(
            csv.DictReader(
                io.TextIOWrapper(z.open("FSD50K.metadata/collection/collection_eval.csv"))
            )
        )
    selected, excluded = [], []
    for r in sorted(rows, key=lambda r: int(r["fname"])):
        fid, labels = r["fname"], r["labels"].split(",")
        meta = info[fid]
        reason = (
            "previous_freesound_parent"
            if fid in known
            else "uploader_linked_to_previous_parent"
            if meta["uploader"] in linked
            else "aircraft_label"
            if any("aircraft" in s.lower() or s == "Helicopter" for s in labels)
            else "uav_metadata_ambiguity"
            if re.search(
                r"\b(drone|drones|quadcopter|quadcoptor|uav|multirotor)\b", json.dumps(meta), re.I
            )
            else None
        )
        if reason:
            excluded.append({"freesound_id": fid, "reason": reason})
        else:
            selected.append(
                dict(
                    freesound_id=fid,
                    labels=labels,
                    **meta,
                    mechanical_confuser=bool(set(labels) & MECHANICAL),
                )
            )
    return (
        selected,
        excluded,
        dict(
            known_parents=len(known),
            known_parents_linked_in_fsd=len(known & all_info.keys()),
            linked_uploaders=len(linked),
            uploader_linkage_complete=False,
            pretraining_overlap_ruled_out=False,
        ),
    )


def wsu_windows(work):
    provenance = read_json(ROOT / "reports/corpus-inventory-v2/wsu-provenance.json")
    headers = read_json(ROOT / "reports/corpus-acquisition-v2/wsu-fixed-view-headers.json")
    docs = provenance["documents"]
    for i, doc in enumerate(docs):
        name = ["0-README.md", "1-LICENSE.txt"][i] if i < 2 else f"{i}-alignment_params.json"
        if file_hash(work / "wsu-docs" / name) != doc["sha256"]:
            raise ValueError("WSU publisher document changed")
    flights = []
    for i, header in enumerate(headers, 2):
        cfg = read_json(work / "wsu-docs" / f"{i}-alignment_params.json")
        seg = cfg.get("segments", [cfg])[0]
        csv_path = work / "wsu-docs" / f"{i}-flight.csv"
        telemetry = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8-sig"))))
        times = np.array([float(r["time(millisecond)"]) / 1000 for r in telemetry])
        heights = np.array([float(r["height_above_takeoff(feet)"]) for r in telemetry])
        origin = times[seg["start_index_csv"]]
        skip = seg["skip_seconds"]
        finish = min(header["duration_seconds"], skip + times[-1] - origin)
        windows = []
        # Device1 correction is zero for all four views. Use all complete 3s
        # grid cells within publisher alignment AND airborne telemetry (>1ft).
        for n in range(int(math.floor((finish - skip) / 3))):
            a, b = skip + 3 * n, skip + 3 * (n + 1)
            samples = (times >= origin + a - skip) & (times <= origin + b - skip)
            if samples.any() and np.all(heights[samples] > 1):
                windows.append(dict(start_frame=round(a * 48000), frames=144000, label="drone"))
        for s in cfg.get("segments", []):
            if s["name"].startswith("no_drone"):
                for n in range(int((s["skip_seconds_end"] - s["skip_seconds"]) // 3)):
                    a = s["skip_seconds"] + n * 3
                    windows.append(
                        dict(
                            start_frame=round(a * 48000),
                            frames=144000,
                            label="non_drone",
                            publisher_segment=s["name"],
                        )
                    )
        flights.append(
            dict(
                header=header,
                alignment_document_sha256=docs[i]["sha256"],
                flight_csv_sha256=file_hash(csv_path),
                flight_csv_url=docs[i]["url"].rsplit("/", 1)[0] + "/" + seg["flight_csv"],
                windows=sorted(windows, key=lambda x: x["start_frame"]),
            )
        )
    return flights


def development_manifest(previous):
    rows = load_manifest(previous / "prepared/manifest.jsonl", require_split=True)
    urban = [r for r in rows if r["source"] == "urbansound8k"]
    categories = defaultdict(set)
    for r in urban:
        categories[r["publisher_class"]].add(r["group_id"])
    chosen = set()
    for category, groups in sorted(categories.items()):
        ordered = sorted(groups, key=lambda g: digest(["confusers-v1-42", category, g]))
        chosen.update(ordered[: int(math.floor(0.75 * len(ordered)))])
    result = []
    for raw in rows:
        row = dict(raw)
        row["previous_split"] = raw["split"]
        if row["source"] == "urbansound8k":
            row["split"] = "train" if row["group_id"] in chosen else "development"
        elif row["split"] == "final":
            row["split"] = "development"
        result.append(row)
    if any(
        len({r["split"] for r in result if r["group_id"] == g}) != 1
        for g in {r["group_id"] for r in result}
    ):
        raise ValueError("Urban parent crossed new partitions")
    return result


def make_plan(args):
    fsd, exclusions, overlap = reserve_fsd(args.work, args.urban_metadata)
    old = development_manifest(args.previous)
    plan = dict(
        version=1,
        selection_without_model_outputs=True,
        previous_manifest_sha256=file_hash(args.previous / "prepared/manifest.jsonl"),
        fsd_metadata_sha256={name: file_hash(args.work / name) for name in METADATA},
        esc50_metadata_sha256=ESC_SHA,
        urbansound_metadata_sha256=URBAN_SHA,
        development_assignment_sha256=digest(old),
        fsd_selected=fsd,
        fsd_excluded=exclusions,
        overlap_audit=overlap,
        wsu_flights=wsu_windows(args.work),
        audrok_records=read_jsonl(ROOT / "data/outdoor-point-recordings-v1.jsonl"),
        audrok_public_files=read_json(args.work / "audrok-archives.json"),
        policy=dict(
            crop="FSD/AuDroK one central min(3s,native length); WSU all supported disjoint3s windows",
            channel="physical channel0, no beamforming or array means",
            rate=16000,
            normalization="none external to the pinned BEATs frontend",
            eligibility="finite nonzero crop >=0.25s; decoded identity duplicates quarantined",
            wsu_positive="publisher alignment + entire 3s airborne telemetry >1ft; presence does not establish audibility",
            wsu_negative="only two explicitly bounded publisher no_drone tail intervals",
            fsd_negative="non-aircraft human foreground labels, metadata UAV exclusion; weak absence evidence",
            audrok_positive="publisher recording label inherited to crop; weak audibility",
            rights="per-recording FSD credits/licenses retained; no raw audio redistributed",
            inference="crop classification, not a continuous alarm experiment",
        ),
    )
    out = args.work / "plan.json"
    if out.exists() and read_json(out) != plan:
        raise ValueError("Reserved cohort changed; preserve the original study")
    write_json(out, plan)
    write_jsonl(args.work / "development-manifest.jsonl", old)
    print(
        json.dumps(
            dict(
                plan_sha256=file_hash(out),
                fsd_recordings=len(fsd),
                fsd_mechanical=sum(r["mechanical_confuser"] for r in fsd),
                fsd_uploaders=len({r["uploader"] for r in fsd}),
                wsu_windows=sum(len(f["windows"]) for f in plan["wsu_flights"]),
                wsu_by_flight=[
                    dict(
                        date=f["header"]["date"],
                        flight=f["header"]["flight_id"],
                        counts=dict(Counter(w["label"] for w in f["windows"])),
                    )
                    for f in plan["wsu_flights"]
                ],
                audrok_records=len(plan["audrok_records"]),
                development_counts=dict(Counter(r["split"] for r in old)),
            )
        ),
        flush=True,
    )


def canonical(args, identifier, audio, rate, metadata):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("Nonfinite or non-mono source crop")
    if len(audio) < rate / 4 or not np.any(audio):
        return None
    divisor = math.gcd(rate, 16000)
    if rate != 16000:
        audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
    if not np.isfinite(audio).all():
        raise ValueError("Nonfinite resampling")
    path = args.work / "prepared/audio" / (identifier + ".wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, 16000, subtype="FLOAT")
    return dict(
        schema_version=1,
        id=identifier,
        path=str(path.resolve()),
        source_version="confusers-v1-reserved-public-sources",
        split="final",
        origin="authentic",
        annotation_status="weak",
        label_scope="publisher presence/foreground label; not human-verified crop audibility or absence",
        feature_input_specification="fixed <=3s crop per reserved confuser-study plan; physical channel0; actual signal passed to native frontend; no external gain/padding",
        events=[],
        ambiguous=[],
        parent_id=None,
        duplicate_review_required=False,
        duration_seconds=len(audio) / 16000,
        sha256=file_hash(path),
        decoded_sha256=hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest(),
        **metadata,
    )


def saved_part(args, name):
    path = args.work / "prepared/parts" / (name + ".json")
    if path.exists():
        part = read_json(path)
        if part["plan_sha256"] != file_hash(args.work / "plan.json"):
            raise ValueError("Prepared part belongs to another plan")
        for r in part["records"]:
            if file_hash(r["path"]) != r["sha256"]:
                raise ValueError("Previously prepared crop changed")
        return part
    return None


def publish_part(args, name, rows, receipts):
    write_json(
        args.work / "prepared/parts" / (name + ".json"),
        dict(plan_sha256=file_hash(args.work / "plan.json"), records=rows, receipts=receipts),
    )
    print(name, len(rows), "crops", flush=True)


def prepare_fsd(args, plan):
    items = read_json(args.work / "fsd-acquisition-plan.json")
    paths = {}
    for item in items:
        path = args.work / "archives" / item["source_id"] / item["filename"]
        receipt = read_json(path.with_name(path.name + ".complete.json"))
        observed = verify(path, item)
        if observed["sha256"] != receipt["sha256"]:
            raise ValueError("Verified FSD volume changed")
        paths[item["filename"]] = path
    z = SplitZip([paths["FSD50K.eval_audio.z01"], paths["FSD50K.eval_audio.zip"]])
    for start in range(0, len(plan["fsd_selected"]), 200):
        name = f"fsd-{start:05d}"
        if saved_part(args, name):
            continue
        rows, receipts = [], []
        for meta in plan["fsd_selected"][start : start + 200]:
            fid = meta["freesound_id"]
            member = "FSD50K.eval_audio/" + fid + ".wav"
            raw = z.read(member)
            with sf.SoundFile(io.BytesIO(raw)) as stream:
                frames, rate, channels = len(stream), stream.samplerate, stream.channels
                if channels != 1 or rate != 44100:
                    raise ValueError("Unexpected FSD native audio format")
                length = min(frames, rate * 3)
                offset = (frames - length) // 2
                stream.seek(offset)
                audio = stream.read(length, dtype="float32")
            row = canonical(
                args,
                "fsd50k-eval-" + fid,
                audio,
                rate,
                dict(
                    source="fsd50k-eval",
                    label="non_drone",
                    original_id="freesound:" + fid,
                    session_id="uploader:" + meta["uploader"],
                    label_evidence="FSD50K human foreground labels, aircraft/UAV metadata excluded",
                    publisher_labels=meta["labels"],
                    mechanical_confuser=meta["mechanical_confuser"],
                    uploader=meta["uploader"],
                    license=meta["license"],
                    title=meta["title"],
                    attribution_url="https://freesound.org/people/"
                    + meta["uploader"]
                    + "/sounds/"
                    + fid
                    + "/",
                    native=dict(
                        member=member,
                        member_sha256=hashlib.sha256(raw).hexdigest(),
                        member_crc32=z.entries[member]["crc32"],
                        sample_rate=rate,
                        channels=channels,
                        frames=frames,
                        crop_start_frame=offset,
                        crop_end_frame=offset + length,
                        whole_volumes_publisher_md5_verified=True,
                    ),
                ),
            )
            if row:
                rows.append(row)
            else:
                receipts.append(dict(id=fid, exclusion="short_or_zero_crop"))
        publish_part(args, name, rows, receipts)


def fetch_range(url, start, length, total, modified=None):
    request = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{start + length - 1}", "Accept-Encoding": "identity"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if (
            response.status != 206
            or response.headers.get("Content-Range")
            != f"bytes {start}-{start + length - 1}/{total}"
        ):
            raise ValueError("Server did not honor exact bounded source range")
        if modified and response.headers.get("Last-Modified") != modified:
            raise ValueError("Native source last-modified changed")
        data = response.read(length + 1)
    if len(data) != length:
        raise ValueError("Truncated source range")
    return data


def prepare_wsu(args, plan):
    for f in plan["wsu_flights"]:
        h = f["header"]
        prefix = fetch_range(h["url"], 0, 65536, h["source_bytes"], h["last_modified"])
        if hashlib.sha256(prefix).hexdigest() != h["prefix_sha256"]:
            raise ValueError("WSU native prefix changed")
        for start in range(0, len(f["windows"]), 20):
            name = f"wsu-{h['date']}-{h['flight_id']}-{start:04d}"
            if saved_part(args, name):
                continue
            windows = f["windows"][start : start + 20]
            first = windows[0]["start_frame"]
            last = windows[-1]["start_frame"] + windows[-1]["frames"]
            if last > h["complete_frames"]:
                raise ValueError("WSU crop crosses incomplete trailing frame")
            raw = fetch_range(
                h["url"],
                h["data_offset"] + first * h["block_align"],
                (last - first) * h["block_align"],
                h["source_bytes"],
                h["last_modified"],
            )
            decoded = (
                np.frombuffer(raw, dtype="<i4").reshape(-1, 6)[:, 0].astype(np.float32)
                / 2147483648.0
            )
            rows = []
            for w in windows:
                offset = w["start_frame"] - first
                identifier = f"wsu-{h['date']}-{h['flight_id']}-{w['start_frame']}"
                row = canonical(
                    args,
                    identifier,
                    decoded[offset : offset + w["frames"]],
                    48000,
                    dict(
                        source="wsu-air3",
                        original_id=h["url"],
                        session_id=h["capture_group"],
                        label=w["label"],
                        capture_date=h["date"],
                        flight_id=h["flight_id"],
                        label_evidence="Publisher aligned airborne telemetry >1ft"
                        if w["label"] == "drone"
                        else "Publisher explicitly bounded no-drone ambient tail",
                        native=dict(
                            url=h["url"],
                            selected_channel=0,
                            channels=6,
                            sample_rate=48000,
                            crop_start_frame=w["start_frame"],
                            crop_end_frame=w["start_frame"] + w["frames"],
                            bounded_range_sha256=hashlib.sha256(raw).hexdigest(),
                            whole_file_checksum_verified=False,
                            trailing_partial_frame_bytes=h["partial_frame_bytes"],
                            complete_frames_only=True,
                        ),
                    ),
                )
                if row:
                    rows.append(row)
            publish_part(
                args,
                name,
                rows,
                [
                    dict(
                        range_first_frame=first,
                        range_end_frame=last,
                        range_sha256=hashlib.sha256(raw).hexdigest(),
                        bytes=len(raw),
                    )
                ],
            )


def prepare_audrok(args, plan):
    files = {r["name"]: r for r in plan["audrok_public_files"]}
    # The size is fixed in the previously published archive audit, alongside the
    # whole-archive reference SHA. Partial reads do not re-verify that SHA.
    lengths = {
        "Renamed_Dresden-II-S-A.zip": 3515538927,
        "Renamed_Dresden-II-S-B.zip": 3168015789,
        "MultiDrone.zip": 200947757,
    }
    for archive, length in lengths.items():
        reader = RangeReader(
            url="https://drive.usercontent.google.com/download?id="
            + files[archive]["id"]
            + "&export=download&confirm=t",
            length=length,
            block_size=4 * 1024 * 1024,
            cache_dir=args.work / "audrok-ranges" / archive,
        )
        with zipfile.ZipFile(reader) as z:
            for meta in [r for r in plan["audrok_records"] if r["source_archive"] == archive]:
                identifier = "audrok-" + digest([archive, meta["member_path"]])[:20]
                if saved_part(args, identifier):
                    continue
                item = z.getinfo(meta["member_path"])
                if item.file_size != meta["member_bytes"] or item.CRC != meta["member_crc32"]:
                    raise ValueError("AuDroK member index changed")
                raw = z.read(item)
                native_sha = hashlib.sha256(raw).hexdigest()
                if meta["member_sha256"] and native_sha != meta["member_sha256"]:
                    raise ValueError("AuDroK previously observed native bytes changed")
                with sf.SoundFile(io.BytesIO(raw)) as stream:
                    if (len(stream), stream.channels, stream.samplerate) != (
                        meta["native_frames"],
                        1,
                        44100,
                    ):
                        raise ValueError("AuDroK native format changed")
                    length_frames = min(len(stream), 3 * stream.samplerate)
                    offset = (len(stream) - length_frames) // 2
                    stream.seek(offset)
                    audio = stream.read(length_frames, dtype="float32")
                row = canonical(
                    args,
                    identifier,
                    audio,
                    44100,
                    dict(
                        source="audrok-dresden",
                        label="drone",
                        original_id=meta["member_path"],
                        session_id=meta["group_id"],
                        label_evidence=meta["label_status"],
                        publisher_drone_code=meta["publisher_drone_code"],
                        publisher_condition_code=meta["publisher_condition_code"],
                        native=dict(
                            member=meta["member_path"],
                            member_sha256=native_sha,
                            member_crc32=item.CRC,
                            archive=archive,
                            sample_rate=44100,
                            channels=1,
                            crop_start_frame=offset,
                            crop_end_frame=offset + length_frames,
                            member_crc_verified=True,
                            whole_archive_checksum_verified_here=False,
                        ),
                    ),
                )
                publish_part(args, identifier, [row] if row else [], [])


def seal(args):
    old = read_jsonl(args.work / "development-manifest.jsonl")
    plan = read_json(args.work / "plan.json")
    expected_parts = {f"fsd-{start:05d}" for start in range(0, len(plan["fsd_selected"]), 200)}
    expected_parts |= {
        f"wsu-{f['header']['date']}-{f['header']['flight_id']}-{start:04d}"
        for f in plan["wsu_flights"]
        for start in range(0, len(f["windows"]), 20)
    }
    expected_parts |= {
        "audrok-" + digest([r["source_archive"], r["member_path"]])[:20]
        for r in plan["audrok_records"]
    }
    actual_parts = {p.stem for p in (args.work / "prepared/parts").glob("*.json")}
    if actual_parts != expected_parts:
        raise ValueError(
            f"Prepared part coverage mismatch: missing={sorted(expected_parts - actual_parts)}; unexpected={sorted(actual_parts - expected_parts)}"
        )
    for name in expected_parts:
        saved_part(args, name)
    candidates = [
        r
        for p in sorted((args.work / "prepared/parts").glob("*.json"))
        for r in read_json(p)["records"]
    ]
    expected = read_json(args.work / "plan.json")
    if not {"fsd50k-eval", "wsu-air3", "audrok-dresden"} <= {r["source"] for r in candidates}:
        raise ValueError("New cohort preparation is incomplete")
    old_hashes = {r["decoded_sha256"] for r in old}
    candidate_labels = defaultdict(set)
    for r in candidates:
        candidate_labels[r["decoded_sha256"]].add(r["label"])
    seen, rows, excluded = set(), [], []
    for r in sorted(candidates, key=lambda r: r["id"]):
        if len(candidate_labels[r["decoded_sha256"]]) > 1:
            excluded.append(dict(id=r["id"], reason="decoded_identity_label_conflict"))
        elif r["decoded_sha256"] in old_hashes or r["decoded_sha256"] in seen:
            excluded.append(dict(id=r["id"], reason="decoded_crop_identity_duplicate"))
        else:
            seen.add(r["decoded_sha256"])
            rows.append(r)
    # One uploader / physical campaign remains a single uncertainty group.
    all_rows = old + rows
    for r, group in zip(all_rows, components(all_rows), strict=True):
        r["group_id"] = group
    write_jsonl(args.work / "prepared/manifest.jsonl", all_rows)
    load_manifest(args.work / "prepared/manifest.jsonl", require_split=True)
    write_json(
        args.work / "prepared/preparation-complete.json",
        dict(
            manifest_sha256=file_hash(args.work / "prepared/manifest.jsonl"),
            plan_sha256=file_hash(args.work / "plan.json"),
            preparation_implementation_sha256=file_hash(__file__),
            new_final_clips=len(rows),
            final_counts=dict(Counter(r["source"] for r in rows)),
            final_groups={
                s: len({r["group_id"] for r in rows if r["source"] == s})
                for s in {r["source"] for r in rows}
            },
            exclusions=excluded,
            reserved_candidates=len(expected["fsd_selected"])
            + len(expected["audrok_records"])
            + sum(len(f["windows"]) for f in expected["wsu_flights"]),
        ),
    )
    print(json.dumps(read_json(args.work / "prepared/preparation-complete.json")), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["plan", "fsd", "wsu", "audrok", "seal"])
    p.add_argument("--work", type=Path, default=Path("work/confuser-study-v1"))
    p.add_argument("--previous", type=Path, default=Path("work/milestone-v1"))
    p.add_argument(
        "--urban-metadata",
        type=Path,
        default=Path("/workspace/drone-detection/work/intake-adapters/metadata/urbansound.csv"),
    )
    args = p.parse_args()
    if args.stage == "plan":
        make_plan(args)
    elif args.stage == "seal":
        seal(args)
    else:
        plan = read_json(args.work / "plan.json")
        {"fsd": prepare_fsd, "wsu": prepare_wsu, "audrok": prepare_audrok}[args.stage](args, plan)


if __name__ == "__main__":
    main()
