"""Reporting guards use small software fixtures, never research evaluations."""

import importlib.util
from pathlib import Path

import pytest

from drone_detector.common import file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "adaptation_report", ROOT / "scripts/report_beats_adaptation.py"
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_saved_running_status_is_interrupted_when_its_process_is_absent(tmp_path, monkeypatch):
    write_json(
        tmp_path / "pipeline-state.json",
        dict(
            status="running",
            child_pid=123,
            accumulated_seconds=7200,
            identity=dict(device="cpu", threads=2),
            stage="four-full-fine-tuning-arms",
        ),
    )
    monkeypatch.setattr(report, "process_active", lambda *_: False)
    (tmp_path / "four-full-fine-tuning-arms.log").write_text(
        '{"epoch":1,"batch":25,"steps":25,"elapsed_seconds":200}\n'
        '{"epoch":1,"batch":50,"steps":50,"elapsed_seconds":400}\n'
        'Resuming exact training batch 0 25 25\n'
        '{"epoch":1,"batch":50,"steps":50,"elapsed_seconds":200}\n'
    )
    result = report.execution(tmp_path, 100)
    assert result["status"] == "interrupted"
    assert result["saved_supervisor_status"] == "running"
    assert result["last_resume_checkpoint_steps"] == 25
    # The resumed clock starts again; it must not halve the earlier timing.
    assert result["first_training_segment_timing"]["seconds_per_batch"] == 8
    monkeypatch.setattr(report, "process_active", lambda *_: True)
    assert report.execution(tmp_path, 100)["status"] == "running"


def artifact(tmp_path, family):
    folder = tmp_path / "wsu-air3/current" / family
    folder.mkdir(parents=True)
    (folder / "head.npz").write_bytes(b"software fixture; not a learned classifier")
    model = dict(
        fold="wsu-air3",
        positive_mix="current",
        plan_sha256="declared-plan",
        train_ids_sha256="declared-train",
        parameters_sha256=file_hash(folder / "head.npz"),
    )
    write_json(folder / "model.json", model)
    write_json(folder / "validation.json", dict(model=model))
    return folder, model


def test_report_rejects_changed_classifier_and_cross_fold_model(tmp_path):
    folder, model = artifact(tmp_path, "frozen")
    args = (tmp_path, "wsu-air3", "current", "frozen", "declared-plan", 9, 6, "declared-train", {"epochs": 3})
    assert report.verified_result(*args)["model"] == model
    (folder / "head.npz").write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="identity changed"):
        report.verified_result(*args)
    model["parameters_sha256"] = file_hash(folder / "head.npz")
    model["fold"] = "audrok-dresden"
    write_json(folder / "model.json", model)
    write_json(folder / "validation.json", dict(model=model))
    with pytest.raises(ValueError, match="identity changed"):
        report.verified_result(*args)


def test_partial_epochs_never_appear_as_completed_fine_tuning(tmp_path):
    folder, model = artifact(tmp_path, "fine-tuned")
    frozen = tmp_path / "wsu-air3/current/frozen"
    write_json(frozen / "model.json", dict(encoder={"fixture": True}))
    recipe = {"epochs": 3}
    training = dict(
        identity=dict(
            recipe=recipe,
            base_encoder={"fixture": True},
            plan_sha256="declared-plan",
            train_ids_sha256="declared-train",
        ),
        history=[dict(epoch=1), dict(epoch=2)],
    )
    write_json(folder / "training-complete.json", training)
    write_json(folder / "validation.json", dict(model=model, training=training))
    with pytest.raises(ValueError, match="complete full-backbone"):
        report.verified_result(
            tmp_path, "wsu-air3", "current", "fine-tuned", "declared-plan", 9, 6,
            "declared-train", recipe,
        )
