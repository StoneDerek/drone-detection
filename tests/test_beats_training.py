import copy
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")

from drone_detector.beats_training import (  # noqa: E402
    TaskModel,
    configure,
    read_state,
    save_state,
    train,
)
from drone_detector.common import file_hash, write_json  # noqa: E402


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.project = torch.nn.Linear(3, 768)
        self.dropout = torch.nn.Dropout(0.3)

    def extract_features(self, waveform):
        output = self.project(waveform[:, :3])
        if self.training and np.random.random() < 0.4:
            output = output + 0.2
        return self.dropout(output).unsqueeze(1), None


def model():
    configure("cpu", seed=42, threads=1)
    head = dict(
        mean=np.zeros(768), scale=np.ones(768), coef=np.ones(768) * 0.01, intercept=np.zeros(1)
    )
    return TaskModel(TinyEncoder(), head)


def records(tmp_path):
    rows = []
    for i in range(8):
        p = tmp_path / f"{i}.wav"
        sf.write(
            p,
            np.random.default_rng(i).normal(0, 0.01, 4000).astype(np.float32),
            16000,
            subtype="FLOAT",
        )
        rows.append(
            dict(
                id=str(i),
                path=str(p),
                sha256=file_hash(p),
                duration_seconds=0.25,
                source="fixture",
                group_id=str(i),
                label="drone" if i % 2 else "non_drone",
            )
        )
    return rows[:6], rows[6:]


def test_mid_epoch_recovery_matches_uninterrupted_weights_and_selection(tmp_path):
    training, selection = records(tmp_path)
    recipe = dict(
        encoder_lr=1e-4, head_lr=1e-3, encoder_weight_decay=0.01, epochs=3, batch_size=2, seed=42
    )
    uninterrupted = model()
    first = train(
        uninterrupted,
        training,
        selection,
        recipe,
        tmp_path / "full",
        {"test": 1},
        "cpu",
        checkpoint_every=2,
    )
    interrupted = model()
    result = train(
        interrupted,
        training,
        selection,
        recipe,
        tmp_path / "resume",
        {"test": 1},
        "cpu",
        checkpoint_every=2,
        max_steps=2,
    )
    assert result["interrupted_for_recovery_test"]
    # A genuinely fresh model and optimizer reconstruct the exact next batch.
    resumed = model()
    second = train(
        resumed,
        training,
        selection,
        recipe,
        tmp_path / "resume",
        {"test": 1},
        "cpu",
        checkpoint_every=2,
    )
    assert first["history"] == second["history"]
    assert first["encoder_state_digest"] == second["encoder_state_digest"]
    assert first["initial_encoder_state_digest"] != first["encoder_state_digest"]
    assert first["optimizer_steps"] == 9
    with pytest.raises(ValueError, match="identity changed"):
        train(model(), training, selection, recipe, tmp_path / "resume", {"test": 2}, "cpu")


def test_pending_checkpoint_recovers_after_data_rename(tmp_path):
    p = tmp_path / "state.pt"
    save_state(p, {"x": torch.tensor([1.0])})
    # Simulate a process dying after the new data rename but before its receipt.
    torch.save({"x": torch.tensor([2.0])}, p)
    write_json(p.with_suffix(".pt.pending.json"), dict(sha256=file_hash(p)))
    assert read_state(p)["x"].item() == 2
    assert not p.with_suffix(".pt.pending.json").exists()
    p.write_bytes(p.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="checksum"):
        read_state(p)


def test_checkpoint_cadence_change_preserves_exact_training_trajectory(tmp_path):
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "checkpoint_cadence", root / "scripts/change_beats_checkpoint_interval.py"
    )
    cadence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cadence)
    training, selection = records(tmp_path)
    recipe = dict(
        encoder_lr=1e-4, head_lr=1e-3, encoder_weight_decay=0.01, epochs=3, batch_size=2, seed=42
    )
    identity = dict(
        test="software fixture; not a research evaluation",
        checkpoint_every_steps=1000,
        training_implementation_sha256=file_hash(root / "src/drone_detector/beats_training.py"),
        driver_sha256=file_hash(root / "scripts/train_beats_adaptation.py"),
    )
    full = train(model(), training, selection, recipe, tmp_path / "full", identity, "cpu")
    folder = tmp_path / "resumed"
    train(model(), training, selection, recipe, folder, identity, "cpu", max_steps=2)
    before = cadence.training_fingerprint(read_state(folder / "latest.pt"))
    assert cadence.change_arm(folder, 1)["training_state_unchanged"]
    assert cadence.training_fingerprint(read_state(folder / "latest.pt")) == before
    assert cadence.change_arm(folder, 1)["training_state_unchanged"]
    target = dict(identity, checkpoint_every_steps=1)
    resumed = train(
        model(), training, selection, recipe, folder, target, "cpu", checkpoint_every=1
    )
    assert resumed["history"] == full["history"]
    assert resumed["encoder_state_digest"] == full["encoder_state_digest"]
    assert resumed["optimizer_steps"] == full["optimizer_steps"] == 9


def test_budget_pause_commits_a_complete_batch_and_can_resume(tmp_path):
    training, selection = records(tmp_path)
    recipe = dict(
        encoder_lr=1e-4, head_lr=1e-3, encoder_weight_decay=0.01, epochs=1, batch_size=2, seed=42
    )
    stop = tmp_path / "pause-requested.json"
    stop.write_text('{"reason":"budget test"}')
    paused = train(
        model(), training, selection, recipe, tmp_path / "run", {"test": 1}, "cpu", stop_path=stop
    )
    assert paused["paused_for_budget_or_user_request"] and paused["steps"] == 1
    assert not (tmp_path / "run/training-complete.json").exists()
    assert read_state(tmp_path / "run/latest.pt")["position"] == 1
    stop.unlink()
    completed = train(
        model(), training, selection, recipe, tmp_path / "run", {"test": 1}, "cpu", stop_path=stop
    )
    assert completed["optimizer_steps"] == 3 and completed["selected_epoch"] == 1


def test_feature_matrix_recovers_a_pending_receipt(tmp_path):
    from drone_detector.beats_training import matrix_metadata, save_matrix

    p = tmp_path / "embeddings.npz"
    save_matrix(p, np.zeros((2, 768), dtype=np.float32), ["a", "b"], {"identity": "one"})
    assert matrix_metadata(p)["identity"] == "one"
    np.savez_compressed(p, x=np.ones((2, 768), dtype=np.float32), ids=np.array(["a", "b"]))
    write_json(p.with_suffix(".pending.json"), dict(sha256=file_hash(p), identity="two"))
    assert matrix_metadata(p)["identity"] == "two"
    assert not p.with_suffix(".pending.json").exists()


def test_waveform_batch_rejects_mixed_lengths(tmp_path):
    from drone_detector.beats_training import load_waveforms

    training, _ = records(tmp_path)
    altered = copy.deepcopy(training[1])
    p = tmp_path / "short.wav"
    sf.write(p, np.ones(4800, dtype=np.float32) * 0.1, 16000, subtype="FLOAT")
    altered.update(path=str(p), sha256=file_hash(p), duration_seconds=0.3)
    with pytest.raises(ValueError, match="Mixed lengths"):
        load_waveforms([training[0], altered], [0, 1], "cpu")


def test_real_native_frontend_and_batch_embeddings_are_preserved():
    path = os.environ.get("BEATS_TRAIN_TEST_ASSETS")
    if not path:
        pytest.skip("Opt-in verification against actual local BEATs assets")
    from drone_detector.beats_encoder import BEATsEncoder

    configure("cpu", threads=1)
    loader = BEATsEncoder(Path(path))
    audio = np.random.default_rng(42).normal(0, 0.01, 48000).astype(np.float32)
    expected = loader.features(audio)
    initial = dict(
        mean=np.zeros(768), scale=np.ones(768), coef=np.ones(768) * 0.001, intercept=np.zeros(1)
    )
    task = TaskModel(loader._load(), initial).eval()
    with torch.inference_mode():
        actual = task.embedding(torch.from_numpy(audio)[None])[0].numpy()
        batched = task.embedding(torch.from_numpy(np.stack([audio, audio]))).numpy()
    np.testing.assert_array_equal(actual.astype(np.float64), expected)
    np.testing.assert_allclose(batched, np.stack([actual, actual]), rtol=2e-5, atol=2e-6)
    assert all(p.requires_grad for p in task.encoder.parameters())


def test_real_training_driver_completes_and_refits_a_head(tmp_path):
    assets = os.environ.get("BEATS_TRAIN_TEST_ASSETS")
    if not assets:
        pytest.skip("Opt-in native training-driver contract; synthetic signals only")
    import importlib.util
    import json
    from types import SimpleNamespace

    from drone_detector.adaptation import cohort_signature
    from drone_detector.beats_encoder import BEATsEncoder
    from drone_detector.common import write_jsonl
    from drone_detector.data import components

    # This checks the complete code path, not acoustic detection performance.
    rows = []
    for i in range(112):
        p = tmp_path / f"signal-{i}.wav"
        audio = np.random.default_rng(i).normal(0, 0.01, 4000).astype(np.float32)
        sf.write(p, audio, 16000, subtype="FLOAT")
        row = dict(
            schema_version=1,
            id=str(i),
            path=str(p),
            source="synthetic-contract",
            source_version="1",
            label_evidence="arbitrary synthetic software-test label",
            label="drone" if i in [1, 3, 5, 106, 107, 110, 111] else "non_drone",
            origin="synthetic",
            annotation_status="unknown",
            events=[],
            ambiguous=[],
            duration_seconds=0.25,
            original_id=str(i),
            session_id=str(i),
            duplicate_review_required=False,
            sha256=file_hash(p),
            split="train" if i < 6 else "development",
        )
        rows.append(row)
    for row, group in zip(rows, components(rows), strict=True):
        row["group_id"] = group
    write_jsonl(tmp_path / "manifest.jsonl", rows)
    recipe = dict(
        epochs=1, batch_size=4, seed=42, encoder_lr=1e-5, head_lr=1e-4, encoder_weight_decay=0.01
    )
    write_json(
        tmp_path / "plan.json",
        dict(
            recipe=dict(cohort_signature=cohort_signature(rows), fine_tuning=recipe),
            folds={
                "wsu-air3": dict(
                    current_positive_train=[str(i) for i in range(6)],
                    selection=[str(i) for i in range(6, 108)],
                    validation=[str(i) for i in range(108, 112)],
                )
            },
        ),
    )
    folder = tmp_path / "wsu-air3/current/frozen"
    folder.mkdir(parents=True)
    np.savez_compressed(
        folder / "head.npz",
        mean=np.zeros(768),
        scale=np.ones(768),
        coef=np.ones(768) * 0.001,
        intercept=np.zeros(1),
    )
    configure("cpu", threads=1)
    base = dict(BEATsEncoder(assets).identity, torch_intraop_threads=1)
    write_json(
        folder / "model.json", dict(encoder=base, parameters_sha256=file_hash(folder / "head.npz"))
    )
    script = Path(__file__).resolve().parents[1] / "scripts/train_beats_adaptation.py"
    spec = importlib.util.spec_from_file_location("native_training_driver_contract", script)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    args = SimpleNamespace(
        work=tmp_path,
        assets=Path(assets),
        device="cpu",
        threads=1,
        checkpoint_every=1000,
        keep_completed_optimizer=False,
    )
    result = driver.run_arm(args, "wsu-air3", "current")
    assert result["model"]["encoder"]["family"] == "beats-task-adapted"
    assert result["training"]["selected_epoch"] == 1
    assert (
        result["training"]["encoder_state_digest"]
        != result["training"]["initial_encoder_state_digest"]
    )
    assert result["validation"]["pooled"]["clips"] == 4
    output = tmp_path / "wsu-air3/current/fine-tuned"
    assert (output / "best.pt").exists() and not (output / "latest.pt").exists()
    assert json.loads((output / "validation.json").read_text())["model"] == result["model"]
