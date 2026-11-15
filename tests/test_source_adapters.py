"""Intake tests focus on label strength, native identity and split contamination."""

import csv

import numpy as np
import pytest
import soundfile as sf

from drone_detector.common import file_hash, read_jsonl, write_jsonl
from drone_detector.data import load_manifest
from drone_detector.source_adapters import (
    dnc_metadata,
    generic_metadata,
    metadata_plan,
    prepare_manifest,
    tut2017_metadata,
    urbansound8k_metadata,
)


def audio(path, seconds=1, frequency=440, channels=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    wave = .1 * np.sin(2 * np.pi * frequency * np.arange(round(16000 * seconds)) / 16000)
    if channels > 1:
        wave = np.repeat(wave[:, None], channels, axis=1)
    sf.write(path, wave, 16000, subtype="PCM_24")
    return file_hash(path)


def urban_csv(path):
    rows = [
        ["123-5-0-0.wav", "123", "0", "1", "1", "1", "5", "engine_idling"],
        ["123-5-0-1.wav", "123", ".5", "1.5", "1", "2", "5", "engine_idling"],
    ]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["slice_file_name", "fsID", "start", "end", "salience", "fold", "classID", "class"])
        writer.writerows(rows)


def test_tut_rename_mapping_preserves_offsets_and_weak_scene_label(tmp_path):
    root = tmp_path / "TUT"
    (root / "evaluation_setup").mkdir(parents=True)
    (root / "meta.txt").write_text("audio/1245.wav\tbeach\tb174\n")
    (root / "evaluation_setup/map.txt").write_text("audio/b174_90_100.wav\taudio/1245.wav\n")
    rows = tut2017_metadata(root, "evaluation")
    row = rows[0]
    assert row["source_interval_seconds"] == [90, 100]
    assert row["original_id"] == "b174"
    assert row["annotation_status"] == "weak"
    assert row["events"] == []
    assert row["split"] == "unassigned"
    assert row["source_recording"] == "audio/b174_90_100.wav"
    assert row["provenance_keys"] == ["tut2017-recording:b174"]
    assert "not an audited" in row["label_evidence"]


def test_tut_requires_rename_mapping_and_matching_parent(tmp_path):
    (tmp_path / "meta.txt").write_text("audio/1245.wav\tbeach\tb174\n")
    with pytest.raises(ValueError, match="require"):
        tut2017_metadata(tmp_path, "evaluation")
    (tmp_path / "evaluation_setup").mkdir()
    (tmp_path / "evaluation_setup/map.txt").write_text("audio/b999_90_100.wav\taudio/1245.wav\n")
    with pytest.raises(ValueError, match="disagree"):
        tut2017_metadata(tmp_path, "evaluation")


def test_urban_parent_across_publisher_folds_remains_one_group(tmp_path):
    csv_path = tmp_path / "metadata.csv"
    urban_csv(csv_path)
    records = urbansound8k_metadata(csv_path)
    original_hashes = []
    for i, row in enumerate(records):
        row["audio_root"] = str(tmp_path)
        original_hashes.append(audio(tmp_path / row["path"], frequency=440 + i * 10, channels=2))
    output = tmp_path / "prepared/manifest.jsonl"
    report = prepare_manifest(records, output)
    prepared = load_manifest(output)
    assert report["provenance_components"] == 1
    assert {row["publisher_fold"] for row in prepared} == {1, 2}
    assert prepared[1]["source_interval_seconds"] == [.5, 1.5]
    assert all(row["channels"] == 2 and row["native_subtype"] == "PCM_24" for row in prepared)
    assert all(row["channel_policy"] == "select_physical_channel_0" for row in prepared)
    assert all(row["channel_index"] == 0 for row in prepared)
    assert [row["sha256"] for row in prepared] == original_hashes
    assert all(row["annotation_status"] == "weak" and row["split"] == "unassigned" for row in prepared)
    with pytest.raises(ValueError, match="Split the manifest"):
        load_manifest(output, require_split=True)


def test_dnc_unknown_session_conservatively_collapses_source(tmp_path):
    audio(tmp_path / "mechanic/a.wav", frequency=440)
    audio(tmp_path / "quiet/b.wav", frequency=100)
    rows = [{**row, "audio_root": str(tmp_path)} for row in dnc_metadata(tmp_path)]
    assert all(row["session_id"] is None for row in rows)
    report = prepare_manifest(rows, tmp_path / "prepared.jsonl")
    assert report["provenance_components"] == 1
    assert report["annotation_counts"] == {"weak": 2}


def test_cross_source_freesound_keys_link_without_changing_source_identity(tmp_path):
    records = []
    for i, (source, keys) in enumerate([("urban", ["freesound:123"]),
                                      ("esc", ["freesound:456"]),
                                      ("derived", ["freesound:123", "freesound:456"]) ]):
        path = f"{source}.wav"
        audio(tmp_path / path, frequency=400 + i * 30)
        meta = tmp_path / f"{source}.jsonl"
        write_jsonl(meta, [{"path": path, "label": "non_drone", "label_evidence": "Publisher class",
                           "original_id": str(i), "session_id": str(i), "provenance_keys": keys}])
        records.extend({**row, "audio_root": str(tmp_path)} for row in generic_metadata(meta, source, "v1"))
    output = tmp_path / "prepared.jsonl"
    assert prepare_manifest(records, output)["provenance_components"] == 1
    prepared = load_manifest(output)
    assert {row["source"] for row in prepared} == {"urban", "esc", "derived"}


def test_related_split_assignments_fail_before_writing(tmp_path):
    csv_path = tmp_path / "metadata.csv"
    urban_csv(csv_path)
    rows = urbansound8k_metadata(csv_path)
    review = {}
    for i, row in enumerate(rows):
        row["audio_root"] = str(tmp_path)
        audio(tmp_path / row["path"], frequency=440 + i * 10)
        review[row["id"]] = {"split": ["train", "development"][i],
                             "provenance_review": "Same parent identified"}
    output = tmp_path / "prepared.jsonl"
    with pytest.raises(ValueError, match="cross assignment splits"):
        prepare_manifest(rows, output, review)
    assert not output.exists()


def test_explicit_review_can_assign_whole_component_without_promoting_labels(tmp_path):
    audio(tmp_path / "a.wav")
    row = dnc_metadata(tmp_path)[0]
    row["audio_root"] = str(tmp_path)
    output = tmp_path / "prepared.jsonl"
    prepare_manifest([row], output,
                     {row["id"]: {"split": "train", "provenance_review": "Whole DNC source retained together"}})
    result = load_manifest(output, require_split=True)[0]
    assert result["annotation_status"] == "weak"
    assert result["events"] == []
    assert result["duplicate_review_required"] is False


def test_native_header_interval_mismatch_is_rejected(tmp_path):
    csv_path = tmp_path / "metadata.csv"
    urban_csv(csv_path)
    row = urbansound8k_metadata(csv_path)[0]
    row["audio_root"] = str(tmp_path)
    audio(tmp_path / row["path"], seconds=.5)
    with pytest.raises(ValueError, match="Header duration"):
        prepare_manifest([row], tmp_path / "prepared.jsonl")


def test_generic_requires_explicit_label_and_rejects_multichannel_mean(tmp_path):
    metadata = tmp_path / "generic.jsonl"
    write_jsonl(metadata, [{"path": "speech.wav", "label_evidence": "Filename says speech"}])
    with pytest.raises(ValueError, match="explicit valid label"):
        generic_metadata(metadata, "polish", "v1")
    write_jsonl(metadata, [{"path": "speech.wav", "label": "drone", "label_evidence": "Publisher drone/speech mixture"}])
    row = generic_metadata(metadata, "polish", "v1")[0]
    assert row["label"] == "drone"
    row["audio_root"] = str(tmp_path)
    audio(tmp_path / "speech.wav", channels=8)
    with pytest.raises(ValueError, match="physical array channel"):
        prepare_manifest([row], tmp_path / "prepared.jsonl")
    row["channel_index"] = 3
    output = tmp_path / "explicit-channel.jsonl"
    prepare_manifest([row], output)
    prepared = load_manifest(output)[0]
    assert prepared["channels"] == 8 and prepared["channel_index"] == 3
    assert prepared["channel_policy"] == "select_physical_channel_3"


@pytest.mark.parametrize("channel", [-1, 2, True, .5])
def test_invalid_physical_channel_is_rejected(tmp_path, channel):
    audio(tmp_path / "a.wav", channels=2)
    row = dnc_metadata(tmp_path)[0]
    row.update({"audio_root": str(tmp_path), "channel_index": channel})
    with pytest.raises(ValueError, match="Physical channel index"):
        prepare_manifest([row], tmp_path / "prepared.jsonl")


def test_metadata_plan_resolves_audio_root_without_requiring_audio(tmp_path):
    csv_path = tmp_path / "metadata.csv"
    urban_csv(csv_path)
    result = metadata_plan({"sources": [{"adapter": "urbansound8k", "metadata": "metadata.csv",
                                          "audio_root": "not-downloaded-yet"}]}, tmp_path)
    assert len(result) == 2
    assert result[0]["audio_root"] == str(tmp_path / "not-downloaded-yet")
    assert not (tmp_path / "not-downloaded-yet").exists()


def test_generic_preserves_timing_and_rejects_path_escape(tmp_path):
    metadata = tmp_path / "generic.jsonl"
    write_jsonl(metadata, [{"path": "../outside.wav", "label": "drone", "label_evidence": "Source label"}])
    with pytest.raises(ValueError, match="Unsafe source-relative"):
        generic_metadata(metadata, "generic", "v1")
    write_jsonl(metadata, [{"path": "a.wav", "label": "drone", "label_evidence": "Source label",
                           "source_interval_seconds": [10, 11]}])
    row = generic_metadata(metadata, "generic", "v1")[0]
    assert row["source_interval_seconds"] == [10, 11]
    assert row["annotation_status"] == "weak"
    assert read_jsonl(metadata)[0]["label"] == "drone"
