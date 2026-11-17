"""Supervision recovery tests; fixtures are not acoustic evaluations."""

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


coordinator = module("coordinate_beats_arms")
launcher = module("launch_beats_adaptation")


def test_completed_result_is_reused_without_executing_or_rewriting_it(tmp_path):
    head = tmp_path / "completed-head"
    head.write_bytes(b"locked classifier fixture")
    before = head.stat().st_mtime_ns
    completed = {"model": {"parameters": "locked"}}
    calls = []

    def execute(fold, mix):
        assert (fold, mix) != coordinator.ARMS[0]
        calls.append((fold, mix))
        return {"model": {"arm": fold + mix}}

    result = coordinator.coordinate(
        lambda fold, mix: completed if (fold, mix) == coordinator.ARMS[0] else None, execute
    )
    assert result["audrok-dresden"]["current"] is completed
    assert calls == coordinator.ARMS[1:]
    assert head.read_bytes() == b"locked classifier fixture" and head.stat().st_mtime_ns == before


def test_corrupt_completed_artifacts_are_not_silently_refitted():
    def invalid(*_):
        raise ValueError("changed completed artifact")

    with pytest.raises(ValueError, match="changed completed"):
        coordinator.coordinate(invalid, lambda *_: pytest.fail("must not refit"))


def test_pause_stops_later_arms_without_publishing_a_complete_study():
    calls = []

    def execute(*arm):
        calls.append(arm)
        return {"status": "paused"}

    assert coordinator.coordinate(lambda *_: None, execute) is None
    assert calls == [coordinator.ARMS[0]]


def states():
    identity = dict(
        plan_sha256="same-plan",
        device="cpu",
        threads=2,
        checkpoint_every_steps=100,
        launcher_sha256=launcher.LEGACY_LAUNCHER_SHA256,
    )
    old = dict(
        identity=identity,
        status="running",
        child_pid=123,
        stage="four-full-fine-tuning-arms",
        accumulated_seconds=98534.66488816802,
        max_wall_hours=44,
    )
    new = dict(identity, launcher_sha256="new-supervisor", arm_coordinator_sha256="new-coordinator")
    return old, new


def test_supervision_migration_preserves_budget_and_every_nonidentity_field():
    old, identity = states()
    original = copy.deepcopy(old)
    new = launcher.migrated_supervision(old, identity, 44)
    assert old == original
    assert new["accumulated_seconds"] == 98534.66488816802
    assert new["status"] == "interrupted"
    assert {k: v for k, v in new.items() if k not in {"status", "identity"}} == {
        k: v for k, v in old.items() if k not in {"status", "identity"}
    }


@pytest.mark.parametrize(
    "change,value",
    [
        ("device", "cuda"),
        ("threads", 1),
        ("checkpoint_every_steps", 1000),
        ("plan_sha256", "other"),
    ],
)
def test_migration_rejects_protocol_or_runtime_changes(change, value):
    old, identity = states()
    identity[change] = value
    with pytest.raises(ValueError, match="protocol"):
        launcher.migrated_supervision(old, identity, 44)


@pytest.mark.parametrize("kwargs", [{"active": True}, {"pause_exists": True}])
def test_migration_respects_existing_writer_or_pause_request(kwargs):
    old, identity = states()
    with pytest.raises(ValueError):
        launcher.migrated_supervision(old, identity, 44, **kwargs)


def test_migration_cannot_extend_or_reset_the_time_budget():
    old, identity = states()
    with pytest.raises(ValueError, match="budget"):
        launcher.migrated_supervision(old, identity, 48)
    old["accumulated_seconds"] = float("nan")
    with pytest.raises(ValueError, match="budget"):
        launcher.migrated_supervision(old, identity, 44)
