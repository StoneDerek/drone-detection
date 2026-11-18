"""Observer control-flow fixtures are not acoustic evaluation results."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from drone_detector.common import file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "completed_arm_observer", ROOT / "scripts/watch_completed_beats_arms.py"
)
watcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watcher)


def test_cli_preserves_the_virtual_environment_interpreter_link(tmp_path, monkeypatch):
    python = tmp_path / "venv-python"
    python.symlink_to(Path(sys.executable).resolve())
    args = [
        "--work",
        "study",
        "--report",
        "report",
        "--private",
        "private",
        "--state",
        "observer.json",
        "--python",
        python.name,
        "--once",
    ]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["observer", *args])
    monkeypatch.setattr(watcher.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(watcher.os, "nice", lambda _: None)

    def observer(options):
        assert options.python == python
        return SimpleNamespace(poll=lambda **_: False)

    monkeypatch.setattr(watcher, "Observer", observer)
    watcher.main()


def test_absent_evaluation_never_starts_analysis(tmp_path, monkeypatch):
    observer = watcher.Observer.__new__(watcher.Observer)
    observer.args = SimpleNamespace(work=tmp_path)
    monkeypatch.setattr(watcher, "verified_result", lambda *_: pytest.fail("not complete"))
    observer.command = lambda *_: pytest.fail("must not analyze partial training")
    assert not observer.observe_arm("audrok-dresden", "broader")


def test_existing_analysis_requires_original_models_inputs_and_private_ledger(tmp_path):
    output, private = tmp_path / "report", tmp_path / "private"
    private.mkdir()
    ledger = private / "paired-predictions.jsonl"
    ledger.write_text("software fixture\n")
    expected = dict(plan_sha256="fixed-plan", compared_validation_rows=7989)
    frozen, adapted = {"model": "locked-frozen"}, {"model": "locked-adapted"}
    summary = dict(
        expected,
        frozen_evaluation=frozen,
        adapted_evaluation=adapted,
        private_paired_ledger_sha256=file_hash(ledger),
    )
    write_json(output / "paired-comparison.json", summary)
    (output / "README.md").write_text("complete software fixture")
    before = (output / "paired-comparison.json").stat().st_mtime_ns
    assert watcher.cached_pair(output, private, expected, frozen, adapted)
    assert (output / "paired-comparison.json").stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="changed"):
        watcher.cached_pair(output, private, dict(expected, plan_sha256="changed"), frozen, adapted)
    ledger.write_text("changed fixture\n")
    with pytest.raises(ValueError, match="changed"):
        watcher.cached_pair(output, private, expected, frozen, adapted)


@pytest.mark.parametrize(
    "status,active,result",
    [
        ("running", True, "running"),
        ("pausing", True, "pausing"),
        ("running", False, "interrupted"),
        ("pausing", False, "interrupted"),
        ("paused", False, "paused"),
        ("failed", False, "failed"),
        ("complete", False, "complete"),
    ],
)
def test_stale_process_state_never_causes_a_training_restart(status, active, result):
    assert watcher.effective_status({"status": status}, active) == result


def test_changed_completed_arm_fails_instead_of_rewriting_analysis(tmp_path):
    folder = tmp_path / "audrok-dresden/current/fine-tuned"
    folder.mkdir(parents=True)
    for name in [
        "validation.json",
        "training-complete.json",
        "model.json",
        "head.npz",
        "best.pt.json",
        "best.pt",
    ]:
        (folder / name).write_bytes(b"software fixture")
    observer = watcher.Observer.__new__(watcher.Observer)
    observer.args = SimpleNamespace(work=tmp_path)
    observer.seen = {"audrok-dresden/current": {"old": "identity"}}
    observer.command = lambda *_: pytest.fail("must preserve changed evidence")
    # The stable-file grace period only avoids observing a just-created JSON.
    import os

    os.utime(folder / "validation.json", (1, 1))
    with pytest.raises(ValueError, match="completed arm changed"):
        observer.observe_arm("audrok-dresden", "current")


@pytest.mark.parametrize(
    "study_status,terminal_first", [("paused", True), ("failed", True), ("running", False)]
)
def test_observer_stops_without_restarting_or_modifying_study(
    tmp_path, monkeypatch, study_status, terminal_first
):
    study = tmp_path / "study"
    study.mkdir()
    for name in ["plan.json", "manifest.jsonl"]:
        (study / name).write_text("software control-flow fixture\n")
    source_state = dict(
        status=study_status, child_pid=123, accumulated_seconds=158400, max_wall_hours=44
    )
    write_json(study / "pipeline-state.json", source_state)
    observer = watcher.Observer.__new__(watcher.Observer)
    observer.args = SimpleNamespace(
        work=study, state=tmp_path / "observer.json", report=tmp_path / "report"
    )
    observer.plan_sha = file_hash(study / "plan.json")
    observer.manifest_sha = file_hash(study / "manifest.jsonl")
    observer.code = {}
    observer.seen = {}
    observer.summaries = {}
    observer.missing_observations = 0
    observer.last_status = None
    observer.observe_arm = lambda *_: False
    commands = []
    observer.command = lambda script, *_: commands.append(script)
    monkeypatch.setattr(watcher, "process_active", lambda *_: False)
    assert observer.poll() == terminal_first
    if not terminal_first:
        assert observer.poll()
    receipt = read_json(observer.args.state)
    assert receipt["status"] == "stopped" and receipt["completed_arms"] == 0
    assert receipt["max_wall_hours"] == 44
    assert read_json(study / "pipeline-state.json") == source_state
    assert not (study / "pause-requested.json").exists()
    assert set(commands) == {"report_beats_adaptation.py"}
