"""Study safeguards; numerical fixtures are not acoustic performance evidence."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from drone_detector.common import digest, file_hash, write_json, write_jsonl
from drone_detector.data import components

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_poland_capture_day_and_speech_never_become_negative():
    prepare = script("prepare_corpus_study")
    for drone, expected in [(1, "train"), (10, "development"), (17, "final")]:
        date, site = prepare.POLAND[drone]
        token = date[2:].replace("-", "")
        name = f"X4_D{drone}_model/OLYMPUSLS11/x4_d{drone}_model_5m_0m_{token}_0006_sekwencja.wav"
        row, reason = prepare.metadata_for(
            "poland-uav-v1", name, Path(f"X4_D{drone}_model.zip"), {}
        )
        assert reason is None and row["label"] == "drone"
        assert row["split"] == expected and row["session_id"] == date + "-" + site
        assert row["speech_marker"]
    row, reason = prepare.metadata_for(
        "poland-uav-v1", "capture/OLYMPUSLS11/x_999999_0001.wav", Path("X4_D1_model.zip"), {}
    )
    assert row is None and reason == "publisher-file-date-conflict"


def test_parallel_microphones_are_not_independent_primary_samples():
    prepare = script("prepare_corpus_study")
    for source, first, second in [
        ("neaptide-v1", "drone_Hover_1.wav", "drone_Hover_2.wav"),
        ("dronenoise-edzell-v3", "event_ev1_M1.wav", "event_ev1_M2.wav"),
    ]:
        accepted, _ = prepare.metadata_for(source, first, Path("source.zip"), {})
        rejected, reason = prepare.metadata_for(source, second, Path("source.zip"), {})
        assert accepted["split"] == "final"
        assert rejected is None and "simultaneous" in reason


def test_weight_mass_equalizes_class_source_and_capture_groups():
    study = script("run_corpus_study")
    rows = [{"label": "drone", "source": "positive-a", "group_id": "a"}] * 6
    rows += [{"label": "drone", "source": "positive-a", "group_id": "b"}]
    rows += [{"label": "drone", "source": "positive-b", "group_id": "c"}] * 2
    rows += [{"label": "non_drone", "source": "negative-a", "group_id": "d"}] * 20
    values = study.weights(rows)

    def mass(**criteria):
        return sum(
            w
            for r, w in zip(rows, values, strict=True)
            if all(r[k] == v for k, v in criteria.items())
        )

    assert np.isclose(mass(label="drone"), mass(label="non_drone"))
    assert np.isclose(mass(source="positive-a"), mass(source="positive-b"))
    assert np.isclose(mass(group_id="a"), mass(group_id="b"))


def test_threshold_uses_only_negative_development_logits_and_handles_ties():
    study = script("run_corpus_study")
    negative = np.r_[np.arange(198), 500.0, 500.0]
    rows = [{"label": "non_drone"} for _ in negative] + [{"label": "drone"}]
    first = study.threshold(np.r_[negative, -1000], rows)
    second = study.threshold(np.r_[negative, 1000], rows)
    assert first == second
    assert np.sum(negative >= first) <= 2
    # Logits remain distinct even where sigmoid rounds both to1.
    assert first > 197


def test_final_clips_are_never_loaded_while_heads_and_cutoffs_are_fitted(tmp_path, monkeypatch):
    study = script("run_corpus_study")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    rows = []
    for split in ["train", "development", "final"]:
        for label in ["drone", "non_drone"]:
            identifier = split + "-" + label
            rows.append(
                {
                    "schema_version": 1,
                    "id": identifier,
                    "path": identifier + ".wav",
                    "source": identifier,
                    "source_version": "fixture",
                    "original_id": identifier,
                    "session_id": identifier,
                    "label": label,
                    "label_evidence": "numerical-test fixture",
                    "split": split,
                    "origin": "synthetic",
                    "duration_seconds": 3,
                    "annotation_status": "weak",
                    "events": [],
                    "ambiguous": [],
                    "sha256": identifier,
                    "duplicate_review_required": False,
                }
            )
    for row, group in zip(rows, components(rows), strict=True):
        row["group_id"] = group
    write_jsonl(prepared / "manifest.jsonl", rows)
    write_json(prepared / "protocol.json", {})
    write_json(
        prepared / "preparation-complete.json",
        {"manifest_sha256": file_hash(prepared / "manifest.jsonl")},
    )
    feature = tmp_path / "features"
    for family in ["spectral", "beats", "clap"]:
        write_json(feature / family / "encoder.json", {"family": "test-" + family})
    seen = []

    def matrix(selected, cache, family):
        seen.extend(r["id"] for r in selected)
        assert all(r["split"] != "final" for r in selected)
        return np.asarray([[int(r["label"] == "drone"), 0.5] for r in selected], dtype=float)

    monkeypatch.setattr(study, "matrix", matrix)
    monkeypatch.setattr(study, "threshold", lambda scores, selected: 0.0)
    args = SimpleNamespace(prepared=prepared, features=feature, output=tmp_path / "models")
    study.freeze(args)
    assert seen and all(not identifier.startswith("final") for identifier in seen)
    assert (args.output / "frozen-models.json").exists()
    assert (prepared / "final-evaluation-authorized.json").exists()
    import pytest

    write_json(feature / "clap/encoder.json", {"family": "changed-test-encoder"})
    with pytest.raises(ValueError, match="differs from the frozen training encoder"):
        study.evaluate(args)
    assert not (args.output / "final-use.json").exists()


def test_faulty_interval_exclusion_uses_actual_crop_boundaries():
    quality = script("finalize_corpus_study")
    row = {"native": {"crop_start_frame": 3500, "crop_end_frame": 6500, "sample_rate": 1000}}
    assert quality.intersects_fault(row, [(2.0, 4.0)])
    assert quality.intersects_fault(row, [(5.0, 9.0)])
    assert not quality.intersects_fault(row, [(0.0, 3.5), (6.5, 10.0)])


def test_gain_copy_crossing_splits_quarantines_both_versions(tmp_path):
    import hashlib
    import soundfile as sf

    quality = script("finalize_corpus_study")
    signal = np.random.default_rng(42).normal(0, 0.1, 4000).astype(np.float32)
    rows = []
    for identifier, split, gain in [("a", "train", 1.0), ("b", "final", 0.5)]:
        path = tmp_path / (identifier + ".wav")
        sf.write(path, signal * gain, 16000, subtype="FLOAT")
        rows.append(
            {
                "id": identifier,
                "path": str(path),
                "sha256": file_hash(path),
                "decoded_sha256": hashlib.sha256((signal * gain).tobytes()).hexdigest(),
                "split": split,
                "label": "non_drone",
                "native": {"member_sha256": identifier},
            }
        )
    accepted, exclusions, pairs = quality.deduplicate(rows)
    assert not accepted
    assert {r["id"] for r in exclusions} == {"a", "b"}
    assert len(pairs) == 1 and pairs[0]["relative_residual"] < 1e-8


def test_changed_finalized_manifest_is_rejected_before_fitting(tmp_path):
    import pytest

    study = script("run_corpus_study")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    write_json(prepared / "preparation-complete.json", {"manifest_sha256": "wrong"})
    (prepared / "manifest.jsonl").write_text("changed\n")
    with pytest.raises(ValueError, match="after quality finalization"):
        study.freeze(SimpleNamespace(prepared=prepared))


def test_public_listing_retains_titles_and_folder_identity():
    acquisition = script("acquire_public_drive_corpus")
    parser = acquisition.FolderParser()
    parser.feed("""<div class="flip-entries">
      <div class="flip-entry" id="entry-folder"><a href="https://drive.google.com/drive/folders/folder">
      <div class="flip-entry-title">source-archives</div></a><div>modified today</div></div>
      <div class="flip-entry" id="entry-file"><a href="https://drive.google.com/file/d/file/view">
      <div class="flip-entry-title">sound &amp; measurement.wav</div></a><div>modified yesterday</div></div></div>""")
    assert parser.rows == [
        {"id": "folder", "name": "source-archives", "folder": True},
        {"id": "file", "name": "sound & measurement.wav", "folder": False},
    ]


def test_incomplete_saved_public_acquisition_is_rejected():
    import pytest

    acquisition = script("acquire_public_drive_corpus")
    with pytest.raises(ValueError, match="incomplete"):
        acquisition.validate_plan({"items": []})


def test_parent_mean_false_positive_rate_does_not_count_repeated_excerpts_as_new_parents():
    study = script("run_corpus_study")
    rows = [{"label": "non_drone", "group_id": "large-parent"} for _ in range(9)]
    rows += [{"label": "non_drone", "group_id": "small-parent"}]
    result = study.metrics(rows, np.r_[np.ones(9), -1], 0.0)
    assert result["negative_clip_FPR"] == 0.9
    assert result["macro_negative_group_FPR"] == 0.5


def test_feature_checkpoint_preserves_numeric_payloads_and_rejects_corruption(tmp_path):
    import pytest

    checkpoint = script("checkpoint_corpus_features")
    source = tmp_path / "source"
    source.mkdir()
    encoder = {"family": "numerical-fixture"}
    specification = {"audio_sha256": "numerical-audio-fixture", "encoder": encoder}
    key = digest(specification)
    payload = source / (key + ".npz")
    np.savez_compressed(payload, vector=np.array([0.125, -2.0, 7.0]))
    write_json(source / "encoder.json", encoder)
    write_json(
        payload.with_suffix(".json"), {"specification": specification, "sha256": file_hash(payload)}
    )
    write_json(
        source / "index.json",
        {"fixture": {"key": key, "audio_sha256": specification["audio_sha256"]}},
    )
    archive = tmp_path / "saved/features.tar.gz"
    checkpoint.pack(source, archive)
    restored = tmp_path / "restored"
    checkpoint.restore(restored, archive)
    assert file_hash(restored / payload.name) == file_hash(payload)
    archive.write_bytes(archive.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="corrupt"):
        checkpoint.restore(tmp_path / "second-restore", archive)


def test_full_corpus_notebook_requires_an_exact_pin_and_has_compilable_cells():
    import pytest

    builder = script("build_corpus_study_notebook")
    with pytest.raises(ValueError, match="full published commit"):
        builder.notebook("main")
    notebook = builder.notebook("a" * 40)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile(cell["source"], "notebook-cell-" + cell["id"], "exec")
    assert any(cell["id"] == "full-corpus-preparation" for cell in notebook["cells"])


def test_incremental_extractor_rescans_parts_added_during_a_long_first_pass(tmp_path, monkeypatch):
    extraction = script("extract_corpus_study_features")
    prepared = tmp_path / "prepared"
    (prepared / "parts").mkdir(parents=True)
    first = {"id": "early", "split": "train", "source": "fixture", "sha256": "one"}
    late = {"id": "late", "split": "train", "source": "fixture", "sha256": "two"}
    write_json(prepared / "parts/early.json", {"records": [first]})

    def feature(row, output, compute, identity, implementation):
        if row["id"] == "early":
            write_json(prepared / "parts/late.json", {"records": [late]})
            write_jsonl(prepared / "manifest.jsonl", [first, late])
            write_json(prepared / "preparation-complete.json", {})
        return row["sha256"], False

    monkeypatch.setattr(extraction, "extractor", lambda *a: (None, {"family": "numerical-test"}))
    monkeypatch.setattr(extraction, "feature_record", feature)
    output = tmp_path / "features"
    extraction.run(
        SimpleNamespace(
            output=output,
            prepared=prepared,
            family="spectral",
            assets=None,
            threads=1,
            device="cpu",
            splits=["train"],
            wait=True,
        )
    )
    from drone_detector.common import read_json

    assert set(read_json(output / "index.json")) == {"early", "late"}
