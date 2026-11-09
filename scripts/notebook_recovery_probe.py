#!/usr/bin/env python3
"""A process-kill/restart check, not an acoustic learning experiment.

The numerical fixture exercises dropout, batch-normalization buffers, AdamW,
scheduler, AMP on CUDA, Python/NumPy/Torch RNG and a resumable data cursor.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from drone_detector import torch_recovery
from drone_detector.torch_recovery import CheckpointStore, capture_state, restore_state, sha256, write_json_verified

TOTAL = 12


def fixture():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(64, 12)).astype(np.float32)
    y = (x[:, :3].sum(axis=1) > 0).astype(np.float32)[:, None]
    return torch.from_numpy(x), torch.from_numpy(y)


def identity(device):
    x, y = fixture()
    return {'purpose': 'numeric recovery verification only; no acoustic evidence',
        'version': 1, 'probe_sha256': sha256(__file__), 'recovery_sha256': sha256(torch_recovery.__file__),
        'runtime_lock_sha256': sha256(Path(__file__).resolve().parents[1] / 'notebooks/runtime/uv.lock'),
        'fixture_sha256': hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest(),
        'torch': torch.__version__, 'numpy': np.__version__, 'device_type': device,
        'seed': 730, 'steps': TOTAL, 'batch_size': 8, 'checkpoint_every_steps': 4,
        'model': 'Linear12x24/BatchNorm/ReLU/Dropout0.2/Linear24x1',
        'optimizer': 'AdamW,lr0.003', 'scheduler': 'ExponentialLR,gamma0.97',
        'sampler': 'one explicit CPU generator; num_workers0; cursor+order saved',
        'amp': device == 'cuda'}


def runtime(device):
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('No usable CUDA GPU. Select a GPU runtime; no silent CPU fallback.')
        # Force a real device operation so an incompatible GPU/driver fails early.
        (torch.ones(8, device='cuda') * 2).sum().item()
        torch.cuda.synchronize()
    return {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
            'device_type': device, 'cuda_runtime': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(0) if device == 'cuda' else None,
            'gpu_capability': list(torch.cuda.get_device_capability(0)) if device == 'cuda' else None}


def components(device):
    random.seed(730)
    np.random.seed(730)
    torch.manual_seed(730)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = torch.nn.Sequential(torch.nn.Linear(12, 24), torch.nn.BatchNorm1d(24),
        torch.nn.ReLU(), torch.nn.Dropout(.2), torch.nn.Linear(24, 1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.97)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda', init_scale=128.)
    return model, optimizer, scheduler, scaler


def worker(root, local, device, crash_after=None):
    environment = runtime(device)
    model, optimizer, scheduler, scaler = components(device)
    store = CheckpointStore(root, local, identity(device))
    order_rng = torch.Generator().manual_seed(319)
    x, y = fixture()
    order = torch.randperm(len(x), generator=order_rng)
    data = {'order': order, 'cursor': 0, 'epoch': 0, 'generator': order_rng.get_state()}
    selection = {'loss_history': [], 'best_loss': float('inf'), 'best_step': None}
    step, resumed_from = 0, None
    saved = store.load_latest()
    if saved is not None:
        step, data, selection = restore_state(saved, model, optimizer, scheduler, scaler)
        order_rng.set_state(data['generator'])
        resumed_from = step
    while step < TOTAL:
        if data['cursor'] + 8 > len(x):
            data['order'] = torch.randperm(len(x), generator=order_rng)
            data['cursor'], data['epoch'] = 0, data['epoch'] + 1
        indices = data['order'][data['cursor']:data['cursor'] + 8]
        # Exercise three RNG families, not just deterministic input selection.
        batch = x[indices] + torch.randn_like(x[indices]) * .015
        batch = batch + float(np.random.normal(scale=.01)) + (random.random() - .5) * .01
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=device == 'cuda', dtype=torch.float16):
            logits = model(batch.to(device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[indices].to(device))
        old_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            raise RuntimeError('Unexpected overflow in the small recovery fixture')
        scheduler.step()
        step += 1
        data['cursor'] += 8
        data['generator'] = order_rng.get_state()
        value = float(loss.detach().cpu())
        selection['loss_history'].append(value)
        if value < selection['best_loss']:
            selection['best_loss'], selection['best_step'] = value, step
        if step % 4 == 0 or step == TOTAL:
            store.save(capture_state(model, optimizer, scheduler, scaler, step, data, selection))
        if step == crash_after:
            # Deliberate hard termination after step5: the latest complete save
            # is step4, so one update must be recomputed by a fresh process.
            os._exit(75)
    write_json_verified(Path(root) / 'worker-result.json', {'completed_steps': step,
        'resumed_from_step': resumed_from, 'runtime': environment, 'rejected': store.rejected,
        'scope': 'Numerical software fixture, not a trained drone detector'})


def tree_error(a, b):
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError('Checkpoint tensor layout differs')
        return float(torch.max(torch.abs(a.to(torch.float64) - b.to(torch.float64)))) if a.numel() else 0.
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max((tree_error(a[key], b[key]) for key in a), default=0.)
    if isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        return max((tree_error(x, y) for x, y in zip(a, b, strict=True)), default=0.)
    if isinstance(a, float):
        return abs(a - b)
    assert a == b
    return 0.


def verify(root, local, device):
    environment = runtime(device)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    earlier_invocation = (root / 'probe-report.json').exists()
    if earlier_invocation:
        previous = json.loads((root / 'probe-report.json').read_text())
        if previous['identity'] != identity(device):
            raise ValueError('Verification identity changed; use a new probe directory')
    script = Path(__file__).resolve()
    command = [sys.executable, str(script), 'worker', '--device', device]
    control = root / 'uninterrupted'
    resumed = root / 'interrupted'
    crash_marker = resumed / 'crash-observed.json'
    if not (control / 'worker-result.json').exists():
        subprocess.run(command + ['--durable', str(control), '--local', str(Path(local) / 'control')], check=True)
    if not crash_marker.exists():
        result = subprocess.run(command + ['--durable', str(resumed), '--local', str(Path(local) / 'crash'), '--crash-after', '5'])
        if result.returncode != 75:
            raise RuntimeError(f'Expected hard-stop75, received{result.returncode}')
        store = CheckpointStore(resumed, Path(local) / 'inspect', identity(device))
        state = store.load_latest()
        if state is None or state['step'] != 4:
            raise AssertionError('Hard stop did not leave the expected step4 recovery point')
        write_json_verified(crash_marker, {'returncode': 75, 'recoverable_step': 4})
    continuation = resumed
    if earlier_invocation:
        # Replay the actual persisted step4 state in a new branch/directory;
        # do not claim success merely from reading a finished step12 file.
        source_store = CheckpointStore(resumed, Path(local) / 'source', identity(device))
        row = next(row for row in source_store._records() if row['step'] == 4)
        if not source_store._verify(row):
            raise ValueError('The original persisted step4 checkpoint changed')
        continuation = root / ('replacement-' + uuid.uuid4().hex)
        continuation.mkdir()
        for name in ['identity.json', row['file'], row['file'][:-3] + '.complete.json']:
            shutil.copyfile(resumed / name, continuation / name)
    subprocess.run(command + ['--durable', str(continuation), '--local', str(Path(local) / 'resume')], check=True)
    a = CheckpointStore(control, Path(local) / 'compare-a', identity(device)).load_latest()
    b = CheckpointStore(continuation, Path(local) / 'compare-b', identity(device)).load_latest()
    error = tree_error(a, b)
    tolerance = 0. if device == 'cpu' else 1e-4
    if error > tolerance:
        raise AssertionError(f'Restored state differs from control: {error} > {tolerance}')
    report = {'passed': True, 'identity': identity(device), 'runtime': environment,
        'maximum_state_error': error, 'tolerance': tolerance, 'verified_steps': TOTAL,
        'hard_stop_after_step': 5, 'checkpoint_before_hard_stop': 4,
        'replayed_checkpoint_from_earlier_invocation': earlier_invocation,
        'continuation_directory': str(continuation),
        'checks': ['model+BN buffers', 'AdamW', 'scheduler', 'AMP scaler when CUDA',
                   'Python/NumPy/Torch RNG', 'explicit sampler+cursor', 'loss history+selection state'],
        'scope': 'Fresh-process hard-stop/recovery verification on a numerical fixture. Not acoustic evidence. Mounted Drive read-back is not independently verified cloud durability; rerun after replacing the runtime.'}
    write_json_verified(root / 'reports' / ('probe-' + uuid.uuid4().hex + '.json'), report)
    write_json_verified(root / 'probe-report.json', report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['worker', 'verify', 'preflight'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--durable', type=Path)
    parser.add_argument('--local', type=Path)
    parser.add_argument('--crash-after', type=int)
    args = parser.parse_args()
    if args.phase == 'preflight':
        print(json.dumps(runtime(args.device), indent=2))
        return
    if args.durable is None:
        parser.error('--durable is required')
    local = args.local or Path(tempfile.gettempdir()) / 'drone-recovery-probe'
    if args.phase == 'worker':
        worker(args.durable, local, args.device, args.crash_after)
    else:
        verify(args.durable, local, args.device)


if __name__ == '__main__':
    main()
