"""Native-front-end, task-specific BEATs adaptation and exact training recovery.

Torch is an optional runtime dependency. This module does not change the frozen
encoder adapter or its existing cache identities. Waveform batches have equal
length, so the official frontend never receives implicit waveform padding.
"""

import hashlib
import fcntl
import json
import os
import random
import time
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn

from .adaptation import batches_by_length, group_weights
from .beats_encoder import BEATsEncoder
from .common import digest, file_hash, read_json, write_json


def configure(device, seed=42, threads=1):
    if device not in {"cpu", "cuda"}:
        raise ValueError("Use an explicit cpu or cuda training runtime")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(threads)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; training does not silently use CPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


class TaskModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder.requires_grad_(True)
        self.register_buffer("mean", torch.tensor(head["mean"], dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(head["scale"], dtype=torch.float32))
        if self.mean.shape != (768,) or self.scale.shape != (768,) or not torch.all(self.scale > 0):
            raise ValueError("Invalid initial TRAIN scaler")
        self.head = nn.Linear(768, 1)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor(head["coef"], dtype=torch.float32)[None])
            self.head.bias.copy_(torch.tensor(head["intercept"], dtype=torch.float32))

    def embedding(self, waveform):
        tokens, padding = self.encoder.extract_features(waveform)
        if padding is not None and padding.any():
            raise ValueError("Unexpected native token masking for exact-length batches")
        return tokens.mean(dim=1)

    def forward(self, waveform):
        return self.head((self.embedding(waveform) - self.mean) / self.scale).squeeze(1)


def load_waveforms(rows, indices, device):
    waves = []
    for i in indices:
        row = rows[i]
        if file_hash(row["path"]) != row["sha256"]:
            raise ValueError("Canonical audio changed during training")
        audio, rate = sf.read(row["path"], dtype="float32")
        if (
            rate != 16000
            or audio.ndim != 1
            or not np.isfinite(audio).all()
            or len(audio) != round(row["duration_seconds"] * 16000)
        ):
            raise ValueError("Audio differs from the declared native crop policy")
        waves.append(audio)
    if len({len(w) for w in waves}) != 1:
        raise ValueError("Mixed lengths require a different batch, not waveform padding")
    return torch.from_numpy(np.stack(waves)).to(device)


def optimizer_for(model, recipe):
    return torch.optim.AdamW(
        [
            dict(
                params=list(model.encoder.parameters()),
                lr=recipe["encoder_lr"],
                weight_decay=recipe["encoder_weight_decay"],
            ),
            dict(params=list(model.head.parameters()), lr=recipe["head_lr"], weight_decay=0),
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def train_step(model, optimizer, waveform, labels, sample_weights, normalizer, mass=9959):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scores = model(waveform)
    terms = nn.functional.binary_cross_entropy_with_logits(scores, labels, reduction="none")
    data_loss = (terms * sample_weights).sum() / normalizer
    loss = data_loss + model.head.weight.square().sum() / (2 * mass)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training objective")
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float((terms * sample_weights).sum().detach()), float(norm.detach())


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def save_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(cpu_tree(state), temporary)
    checksum = file_hash(temporary)
    pending = path.with_suffix(path.suffix + ".pending.json")
    write_json(pending, dict(sha256=checksum))
    temporary.replace(path)
    receipt = path.with_suffix(path.suffix + ".json")
    receipt_temporary = receipt.with_suffix(".partial")
    write_json(receipt_temporary, dict(sha256=checksum))
    receipt_temporary.replace(receipt)
    pending.unlink(missing_ok=True)


def read_state(path):
    path = Path(path)
    checksum = file_hash(path)
    receipt = path.with_suffix(path.suffix + ".json")
    pending = path.with_suffix(path.suffix + ".pending.json")
    if pending.exists() and read_json(pending)["sha256"] == checksum:
        write_json(receipt, dict(sha256=checksum))
        pending.unlink()
    if not receipt.exists() or checksum != read_json(receipt)["sha256"]:
        raise ValueError("Training checkpoint checksum changed")
    return torch.load(path, map_location="cpu", weights_only=True)


def save_matrix(path, x, ids, metadata):
    path = Path(path)
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as f:
        np.savez_compressed(f, x=x, ids=np.asarray(ids))
    meta = dict(metadata, sha256=file_hash(temporary))
    pending = path.with_suffix(".pending.json")
    write_json(pending, meta)
    temporary.replace(path)
    receipt = path.with_suffix(".json")
    temp_receipt = receipt.with_suffix(".partial")
    write_json(temp_receipt, meta)
    temp_receipt.replace(receipt)
    pending.unlink(missing_ok=True)


def matrix_metadata(path):
    path = Path(path)
    checksum = file_hash(path)
    pending = path.with_suffix(".pending.json")
    receipt = path.with_suffix(".json")
    if pending.exists() and read_json(pending)["sha256"] == checksum:
        write_json(receipt, read_json(pending))
        pending.unlink()
    meta = read_json(receipt)
    if meta["sha256"] != checksum:
        raise ValueError("Embedding matrix checksum changed")
    return meta


def random_state():
    algorithm, keys, position, has_gauss, cached = np.random.get_state()
    return dict(
        torch_cpu=torch.get_rng_state(),
        python=random.getstate(),
        numpy=dict(
            algorithm=algorithm,
            keys=keys.tolist(),
            position=position,
            has_gauss=has_gauss,
            cached=cached,
        ),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_random(state):
    torch.set_rng_state(state["torch_cpu"])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state(
        (
            n["algorithm"],
            np.asarray(n["keys"], dtype=np.uint32),
            n["position"],
            n["has_gauss"],
            n["cached"],
        )
    )
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def state_digest(module):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def selection_loss(model, rows, device, batch_size):
    model.eval()
    mass = group_weights(rows)
    scores = np.zeros(len(rows), dtype=np.float64)
    with torch.inference_mode():
        for batch in batches_by_length(rows, batch_size, 42, 0, shuffle=False):
            scores[batch] = model(load_waveforms(rows, batch, device)).cpu().numpy()
    y = np.array([r["label"] == "drone" for r in rows], dtype=float)
    loss = float(np.dot(mass, np.logaddexp(0, scores) - y * scores) / mass.sum())
    if not np.isfinite(loss):
        raise ValueError("Nonfinite model-selection objective")
    return loss


def _train(
    model,
    training,
    selection,
    recipe,
    output,
    identity,
    device,
    checkpoint_every=1000,
    max_steps=None,
    stop_path=None,
):
    """Resume exact next batch; max_steps is exclusively for recovery tests."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Training run identity changed")
    write_json(identity_path, identity)
    if (output / "training-complete.json").exists():
        completed = read_json(output / "training-complete.json")
        if file_hash(output / "best.pt") != completed["best_parameters_sha256"]:
            raise ValueError("Completed adapted parameters changed")
        return completed
    latest = output / "latest.pt"
    best = output / "best.pt"
    optimizer = optimizer_for(model, recipe)
    epoch, position, steps, history = 0, 0, 0, []
    best_loss = float("inf")
    initial_digest = state_digest(model.encoder)
    initial_reference = None
    if latest.exists():
        state = read_state(latest)
        if state["identity"] != identity:
            raise ValueError("Resume checkpoint belongs to another run")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        epoch, position, steps = state["epoch"], state["position"], state["steps"]
        history, best_loss = state["history"], state["best_loss"]
        initial_digest, initial_reference = state["initial_digest"], state["initial_reference"]
        restore_random(state["rng"])
        print("Resuming exact training batch", epoch, position, steps, flush=True)
    if initial_reference is None:
        initial_reference = selection_loss(model, selection, device, recipe["batch_size"])
        print("Initial selection weighted log-loss", initial_reference, flush=True)
    weights = torch.tensor(group_weights(training), dtype=torch.float32, device=device)
    labels = torch.tensor(
        [r["label"] == "drone" for r in training], dtype=torch.float32, device=device
    )
    started = time.monotonic()

    def checkpoint(next_epoch, next_position):
        save_state(
            latest,
            dict(
                identity=identity,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=next_epoch,
                position=next_position,
                steps=steps,
                history=history,
                best_loss=best_loss,
                initial_digest=initial_digest,
                initial_reference=initial_reference,
                rng=random_state(),
            ),
        )

    while epoch < recipe["epochs"]:
        batches = batches_by_length(training, recipe["batch_size"], recipe["seed"], epoch)
        normalizer = float(weights.sum()) / len(batches)
        for batch_no in range(position, len(batches)):
            batch = batches[batch_no]
            loss_sum, norm = train_step(
                model,
                optimizer,
                load_waveforms(training, batch, device),
                labels[batch],
                weights[batch],
                normalizer,
            )
            steps += 1
            pause_requested = stop_path is not None and Path(stop_path).exists()
            if steps % 25 == 0:
                print(
                    json.dumps(
                        dict(
                            epoch=epoch + 1,
                            batch=batch_no + 1,
                            batches=len(batches),
                            steps=steps,
                            batch_weighted_loss_sum=loss_sum,
                            gradient_norm=norm,
                            elapsed_seconds=time.monotonic() - started,
                        )
                    ),
                    flush=True,
                )
            if (
                steps % checkpoint_every == 0
                or (max_steps is not None and steps >= max_steps)
                or pause_requested
            ):
                checkpoint(epoch, batch_no + 1)
            if pause_requested:
                return dict(
                    paused_for_budget_or_user_request=True,
                    steps=steps,
                    completed_epochs=epoch,
                    next_batch=batch_no + 1,
                )
            if max_steps is not None and steps >= max_steps:
                return dict(interrupted_for_recovery_test=True, steps=steps)
        value = selection_loss(model, selection, device, recipe["batch_size"])
        history.append(dict(epoch=epoch + 1, selection_weighted_logloss=value, steps=steps))
        if value < best_loss:
            best_loss = value
            save_state(
                best,
                dict(
                    encoder=model.encoder.state_dict(),
                    identity=identity,
                    training_head=model.head.state_dict(),
                    mean=model.mean,
                    scale=model.scale,
                    epoch=epoch + 1,
                    selection_loss=value,
                ),
            )
        epoch += 1
        position = 0
        checkpoint(epoch, 0)
        print("Completed full epoch", json.dumps(history[-1]), flush=True)
    selected = read_state(best)
    model.encoder.load_state_dict(selected["encoder"], strict=True)
    adapted_digest = state_digest(model.encoder)
    if adapted_digest == initial_digest:
        raise ValueError("Selected model has no encoder weight updates")
    result = dict(
        identity=identity,
        selected_epoch=selected["epoch"],
        selected_selection_loss=selected["selection_loss"],
        initial_reference_loss=initial_reference,
        history=history,
        optimizer_steps=steps,
        encoder_state_digest=adapted_digest,
        initial_encoder_state_digest=initial_digest,
        best_parameters_sha256=file_hash(best),
        trainable_encoder_parameters=sum(
            p.numel() for p in model.encoder.parameters() if p.requires_grad
        ),
        training_rows=len(training),
        all_selected_crops_visited_each_epoch=True,
        implementation_sha256=file_hash(__file__),
    )
    write_json(output / "training-complete.json", result)
    return result


def train(
    model,
    training,
    selection,
    recipe,
    output,
    identity,
    device,
    checkpoint_every=1000,
    max_steps=None,
    stop_path=None,
):
    lock_dir = Path(tempfile.gettempdir()) / "drone-beats-training-locks"
    lock_dir.mkdir(exist_ok=True)
    name = hashlib.sha256(str(Path(output).resolve()).encode()).hexdigest()
    # This serializes writers on one runtime. A Drive project must still have
    # only one active notebook/runtime writer; Drive is not a distributed lock.
    with (lock_dir / name).open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another training writer is active for this run") from error
        return _train(
            model,
            training,
            selection,
            recipe,
            output,
            identity,
            device,
            checkpoint_every=checkpoint_every,
            max_steps=max_steps,
            stop_path=stop_path,
        )


def build_model(assets, head_path, device):
    loader = BEATsEncoder(assets, device)
    with np.load(head_path, allow_pickle=False) as z:
        head = {k: z[k].copy() for k in z.files}
    model = TaskModel(loader._load(), head).to(device)
    return model, loader.identity


def embed_selected(model, rows, device, batch_size, stop_path=None):
    model.eval()
    x = np.zeros((len(rows), 768), dtype=np.float32)
    batches = batches_by_length(rows, batch_size, 42, 0, shuffle=False)
    with torch.inference_mode():
        for count, batch in enumerate(batches, 1):
            if stop_path is not None and Path(stop_path).exists():
                raise InterruptedError("Embedding extraction paused; selected encoder is preserved")
            x[batch] = model.embedding(load_waveforms(rows, batch, device)).cpu().numpy()
            if count % 100 == 0:
                print("Adapted embedding batches", count, "/", len(batches), flush=True)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite adapted representations")
    return x


def run_identity(base, recipe, fold, mix, plan_sha, device, threads, train_ids):
    return dict(
        base_encoder=base,
        recipe=recipe,
        held_source=fold,
        positive_mix=mix,
        plan_sha256=plan_sha,
        train_ids_sha256=digest(train_ids),
        device=device,
        threads=threads,
        precision="float32",
        torch_version=str(torch.__version__),
        training_implementation_sha256=file_hash(__file__),
    )
