import copy
import shutil

import pytest

from drone_detector import data
from drone_detector.common import digest, file_hash, read_json, write_json, write_jsonl
from drone_detector.data import audit, components, load_manifest, split_manifest, validate_record
from drone_detector.evaluation import evaluate, tune
from drone_detector.model import load_model, train


def test_split_components_and_reserved_source(prepared):
    rows = prepared["rows"]
    for group in {r["group_id"] for r in rows}:
        assert len({r["split"] for r in rows if r["group_id"] == group}) == 1
    assert all(r["split"] == "final" for r in rows if r["source"] == "synthetic-generator-2")


def test_raw_decoded_parent_and_missing_provenance_group_transitively(prepared):
    rows = copy.deepcopy(prepared["rows"][:3])
    for i, row in enumerate(rows):
        row.update(id=str(i), session_id=str(i), original_id=str(i),
                   sha256=str(i), decoded_sha256=str(i), gain_normalized_sha256=str(i))
    rows[1]["sha256"] = rows[0]["sha256"]
    rows[2]["parent_id"] = rows[1]["id"]
    assert len(set(components(rows))) == 1
    rows[1]["sha256"] = "different"
    rows[2]["parent_id"] = None
    assert len(set(components(rows))) == 3
    rows[0]["session_id"] = None
    assert len(set(components(rows))) == 1


def test_manifest_rejects_cross_split_derivatives(prepared, tmp_path):
    rows = copy.deepcopy(prepared["rows"])
    row = rows[0]
    row["split"] = "development" if row["split"] == "train" else "train"
    path = tmp_path / "leak.jsonl"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="leaks"):
        load_manifest(path, True)


def test_empty_manifest_not_success(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.touch()
    with pytest.raises(ValueError, match="no audio"):
        load_manifest(path)


@pytest.mark.parametrize("change", [
    {"events": [[-1, 2]]}, {"events": [[1, 50]]},
    {"annotation_status": "verified_negative", "events": [[1, 2]]},
    {"label": "drone", "annotation_status": "verified_temporal", "events": []},
    {"events": [[1, 4]], "ambiguous": [[2, 3]]},
])
def test_bad_temporal_labels(prepared, change):
    row = copy.deepcopy(prepared["rows"][0])
    row.update(change)
    with pytest.raises(ValueError):
        validate_record(row)


def test_provenance_review_required(prepared, tmp_path):
    rows = copy.deepcopy(prepared["rows"])
    rows[0]["duplicate_review_required"] = True
    path = tmp_path / "unreviewed.jsonl"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="provenance"):
        load_manifest(path, True)


def test_training_and_tuning_never_decode_final_audio(prepared, tmp_path, monkeypatch):
    accessed = []
    actual = data.checked_audio

    def recording(row, *args, **kwargs):
        assert row["split"] != "final"
        accessed.append(row["split"])
        return actual(row, *args, **kwargs)

    monkeypatch.setattr("drone_detector.model.checked_audio", recording)
    monkeypatch.setattr("drone_detector.evaluation.checked_audio", recording)
    train(prepared["manifest"], tmp_path / "model", "relative_mel")
    tune(tmp_path / "model", prepared["manifest"], prepared["contract"], tmp_path / "frozen")
    assert set(accessed) == {"train", "development"}


def test_audio_hash_changed_rejected(prepared, tmp_path):
    row = copy.deepcopy(prepared["rows"][0])
    original = prepared["manifest"].parent / row["path"]
    copied = tmp_path / "changed.wav"
    shutil.copyfile(original, copied)
    with copied.open("ab") as handle:
        handle.write(b"changed")
    row["path"] = "changed.wav"
    with pytest.raises(ValueError, match="changed since audit"):
        data.checked_audio(row, tmp_path / "manifest.jsonl")


def test_artifact_tampering_and_missing_statistics(prepared, tmp_path):
    target = tmp_path / "model"
    shutil.copytree(prepared["bundle"], target)
    with (target / "parameters.npz").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_model(target)
    (target / "parameters.npz").unlink()
    with pytest.raises(FileNotFoundError):
        load_model(target)


def test_class_mapping_cannot_be_silently_swapped(prepared, tmp_path):
    target = tmp_path / "model"
    shutil.copytree(prepared["bundle"], target)
    meta = read_json(target / "model.json")
    meta["class_mapping"] = {"0": "drone", "1": "non_drone"}
    meta["model_id"] = digest({k: v for k, v in meta.items() if k != "model_id"})
    write_json(target / "model.json", meta)
    with pytest.raises(ValueError, match="class mapping"):
        load_model(target)


def test_untuned_inference_refused(prepared):
    with pytest.raises(ValueError, match="Tune and freeze"):
        load_model(prepared["raw"], require_frozen=True)


def test_all_verified_training_records_used(prepared):
    metadata = prepared["model"].metadata
    ids = {r["id"] for r in prepared["rows"] if r["split"] == "train"}
    assert {r["id"] for r in metadata["training"]["records"]} == ids
    assert metadata["training"]["window_counts"]["1"] > 0


def test_final_comparison_once_and_explicit_reuse(prepared, tmp_path):
    first = evaluate([prepared["bundle"]], prepared["manifest"], "final", tmp_path / "first")
    assert len(first["results"]) == 1
    assert not first["results"][0]["assessment"]["reliability_claim_supported"]
    with pytest.raises(ValueError, match="already evaluated"):
        evaluate([prepared["bundle"]], prepared["manifest"], "final", tmp_path / "second")
    reused = evaluate([prepared["bundle"]], prepared["manifest"], "final", tmp_path / "reuse", True)
    assert reused["final_reused"]
    assert "REUSED" in reused["results"][0]["assessment"]["status"]


def test_audit_without_annotations_does_not_guess_labels(prepared, tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    row = prepared["rows"][0]
    shutil.copyfile(prepared["manifest"].parent / row["path"], root / "yes-drone.wav")
    manifest = tmp_path / "inventory.jsonl"
    audit(root, manifest)
    result = load_manifest(manifest)[0]
    assert result["label"] == "uncertain" and result["origin"] == "unknown"
    assert result["duplicate_review_required"]
    with pytest.raises(ValueError, match="Too few"):
        split_manifest(manifest, tmp_path / "split.jsonl")


def test_manifest_model_binding(prepared, tmp_path):
    rows = copy.deepcopy(prepared["rows"])
    rows[0]["environment"] = "changed metadata"
    manifest = tmp_path / "different.jsonl"
    write_jsonl(manifest, rows)
    assert file_hash(manifest) != file_hash(prepared["manifest"])
    with pytest.raises(ValueError, match="manifest mismatch"):
        evaluate([prepared["bundle"]], manifest, "development", tmp_path / "invalid")
