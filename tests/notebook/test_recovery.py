"""Exercise state and storage failure behavior, separately from acoustic results."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# The normal inference environment has no PyTorch dependency. A dedicated CI
# job installs the notebook CPU lock and must execute these tests without skips.
torch = pytest.importorskip('torch')

from drone_detector.torch_recovery import (  # noqa: E402 (optional dependency gate above)
    CheckpointConflict, CheckpointStore, capture_state, restore_state,
)

ROOT = Path(__file__).resolve().parents[2]


def objects():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    model[1].eval()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.9)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    model(torch.randn(4, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    return model, optimizer, scheduler, scaler


def state(step=1):
    return {'schema_version': 1, 'step': step, 'value': torch.tensor([step])}


def test_snapshot_is_immutable_and_restores_module_modes():
    model, optimizer, scheduler, scaler = objects()
    snapshot = capture_state(model, optimizer, scheduler, scaler, 1,
                             {'cursor': 4}, {'best_step': 1})
    original = snapshot['model']['0.weight'].clone()
    moment = snapshot['optimizer']['state'][0]['exp_avg'].clone()
    with torch.no_grad():
        model[0].weight.add_(1)
        optimizer.state[model[0].weight]['exp_avg'].add_(1)
    assert torch.equal(snapshot['model']['0.weight'], original)
    assert torch.equal(snapshot['optimizer']['state'][0]['exp_avg'], moment)
    model.eval()
    step, data, selection = restore_state(snapshot, model, optimizer, scheduler, scaler)
    assert step == 1 and data['cursor'] == 4 and selection['best_step'] == 1
    assert model.training and model[0].training and not model[1].training
    assert torch.equal(model[0].weight, original)


def test_partial_files_and_corrupt_newest_fall_back(tmp_path):
    durable, local = tmp_path / 'drive', tmp_path / 'local'
    writer = CheckpointStore(durable, local, {'config': 'a'})
    first = writer.save(state(1))
    second = writer.save(state(2))
    (durable / second['file']).write_bytes(b'truncated')
    (durable / 'step-0000000003.partial-abc').write_bytes(b'partial')
    (durable / 'step-0000000004.complete.json').write_text('{')
    reader = CheckpointStore(durable, local, {'config': 'a'})
    assert reader.load_latest()['step'] == 1
    assert reader.parent == first['file'] and reader.rejected
    # Continue from the verified backup rather than getting stuck on bad bytes.
    reader.save(state(3))
    assert CheckpointStore(durable, local, {'config': 'a'}).load_latest()['step'] == 3


def test_identity_and_stale_writer_are_rejected(tmp_path):
    writer = CheckpointStore(tmp_path / 'drive', tmp_path / 'local', {'config': 'a'})
    stale = CheckpointStore(writer.durable, writer.local, {'config': 'a'})
    writer.save(state())
    with pytest.raises(CheckpointConflict, match='writer advanced'):
        stale.save(state(2))
    with pytest.raises(CheckpointConflict, match='identity changed'):
        CheckpointStore(writer.durable, writer.local, {'config': 'b'})


def test_manifest_cannot_point_outside_store(tmp_path):
    store = CheckpointStore(tmp_path / 'drive', tmp_path / 'local', {'config': 'a'})
    row = store.save(state())
    path = store.durable / (row['file'][:-3] + '.complete.json')
    row['file'] = '../elsewhere.pt'
    path.write_text(json.dumps(row))
    assert store.load_latest() is None
    assert store.rejected


def test_visible_divergent_branches_are_rejected(tmp_path):
    # Simulate two complete first saves becoming visible together (e.g. delayed
    # mount visibility). This is detection, not a distributed locking claim.
    a = CheckpointStore(tmp_path / 'a', tmp_path / 'local', {'config': 'a'})
    b = CheckpointStore(tmp_path / 'b', tmp_path / 'local', {'config': 'a'})
    a.save(state(1))
    b.save(state(2))
    for member in b.durable.glob('step-*'):
        (a.durable / member.name).write_bytes(member.read_bytes())
    with pytest.raises(CheckpointConflict, match='Divergent'):
        CheckpointStore(a.durable, a.local, {'config': 'a'}).load_latest()


def test_hard_stop_and_replay_persisted_checkpoint_in_new_process(tmp_path):
    command = [sys.executable, str(ROOT / 'scripts/notebook_recovery_probe.py'),
               'verify', '--device', 'cpu', '--durable', str(tmp_path / 'drive'),
               '--local', str(tmp_path / 'local')]
    environment = {**os.environ, 'PYTHONPATH': str(ROOT / 'src')}
    for invocation in range(2):
        result = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                                capture_output=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads((tmp_path / 'drive/probe-report.json').read_text())
        assert report['passed'] and report['maximum_state_error'] == 0
        assert report['replayed_checkpoint_from_earlier_invocation'] == bool(invocation)
        worker = json.loads((Path(report['continuation_directory']) / 'worker-result.json').read_text())
        assert worker['resumed_from_step'] == 4
