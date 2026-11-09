"""Append-only, verified checkpoints for our own PyTorch training processes.

Optional PyTorch runtime; inference bundles remain numeric NPZ/JSON. A mounted
Drive read-back checks filesystem integrity, not independent cloud durability.
"""

import errno
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import uuid
from collections import defaultdict
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

NAME = re.compile(r'step-\d{10}-[0-9a-f]{64}\.pt')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _flush(stream):
    stream.flush()
    try:
        os.fsync(stream.fileno())
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}:
            raise


def write_json_verified(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json(value)
    temporary = path.with_name(path.name + '.partial-' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        stream.write(payload)
        _flush(stream)
    os.replace(temporary, path)
    if path.read_bytes() != payload:
        raise ValueError('JSON read-back differs after storage write')


def _freeze(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        result = OrderedDict() if isinstance(value, OrderedDict) else {}
        result.update((key, _freeze(item)) for key, item in value.items())
        if hasattr(value, '_metadata'):
            result._metadata = _freeze(value._metadata)
        return result
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, list):
        return [_freeze(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Unsupported checkpoint state type: {type(value).__name__}')


def capture_state(model, optimizer, scheduler, scaler, step, data_state, selection_state):
    numpy_rng = np.random.get_state()
    return _freeze({
        'schema_version': 1, 'step': step, 'model': model.state_dict(),
        'module_training': {name: module.training for name, module in model.named_modules()},
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
        'data_state': data_state, 'selection_state': selection_state,
        'rng': {'python': random.getstate(), 'torch': torch.get_rng_state(),
                'numpy': {'algorithm': numpy_rng[0], 'keys': torch.tensor(numpy_rng[1].astype(np.int64)),
                          'position': numpy_rng[2], 'has_gaussian': numpy_rng[3], 'gaussian': numpy_rng[4]},
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
    })


def restore_state(state, model, optimizer, scheduler, scaler):
    if state['schema_version'] != 1:
        raise ValueError('Unsupported training checkpoint schema')
    model.load_state_dict(state['model'])
    modules = dict(model.named_modules())
    if set(modules) != set(state['module_training']):
        raise ValueError('Checkpoint module structure changed')
    for name, training in state['module_training'].items():
        modules[name].training = training
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    random.setstate(state['rng']['python'])
    numpy_rng = state['rng']['numpy']
    np.random.set_state((numpy_rng['algorithm'], numpy_rng['keys'].numpy().astype(np.uint32),
                         numpy_rng['position'], numpy_rng['has_gaussian'], numpy_rng['gaussian']))
    torch.set_rng_state(state['rng']['torch'])
    if state['rng']['cuda']:
        if not torch.cuda.is_available() or len(state['rng']['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA checkpoint needs the same number of visible CUDA devices')
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state['step'], state['data_state'], state['selection_state']


class CheckpointConflict(ValueError):
    pass


class CheckpointStore:
    """Single-writer store; immutable complete records, no mutable latest pointer.

    Always call load_latest before resuming. Parent checks catch stale writers;
    fork detection rejects visible divergent branches. A mounted filesystem is
    not a distributed lock service, so simultaneous runtimes remain unsupported.
    """

    def __init__(self, durable, local, identity):
        self.durable, self.local = Path(durable), Path(local)
        self.identity = identity
        self.identity_hash = hashlib.sha256(_json(identity)).hexdigest()
        self.parent, self.step, self.invalid = None, -1, set()
        self.rejected = []
        self.durable.mkdir(parents=True, exist_ok=True)
        self.local.mkdir(parents=True, exist_ok=True)
        path = self.durable / 'identity.json'
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise CheckpointConflict('Run identity changed; use a new run directory')
        else:
            write_json_verified(path, identity)

    def _records(self):
        rows = []
        for path in sorted(self.durable.glob('step-*.complete.json')):
            try:
                row = json.loads(path.read_text())
                name = row['file']
                if not NAME.fullmatch(name) or path.name != name[:-3] + '.complete.json':
                    raise ValueError('Invalid checkpoint member path')
                if row['identity_sha256'] != self.identity_hash or row['schema_version'] != 1:
                    raise CheckpointConflict('Complete checkpoint has a different run identity/schema')
                if row['step'] < 0 or row['step'] != int(name.split('-')[1]):
                    raise ValueError('Invalid checkpoint step')
                if (not isinstance(row['bytes'], int) or row['bytes'] <= 0
                        or row['sha256'] != name[16:-3]
                        or (row['parent'] is not None and (not isinstance(row['parent'], str)
                            or not NAME.fullmatch(row['parent'])
                            or int(row['parent'].split('-')[1]) >= row['step']))):
                    raise ValueError('Invalid checkpoint metadata')
                rows.append(row)
            except CheckpointConflict:
                raise
            except (ValueError, KeyError, TypeError) as error:
                self.rejected.append({'manifest': path.name, 'reason': str(error)})
        return [row for row in rows if row['file'] not in self.invalid]

    def _verify(self, row):
        path = self.durable / row['file']
        valid = (path.is_file() and path.resolve().parent == self.durable.resolve()
                 and path.stat().st_size == row['bytes'] and sha256(path) == row['sha256'])
        if not valid:
            self.invalid.add(row['file'])
            self.rejected.append({'file': row['file'], 'reason': 'Missing or corrupt checkpoint bytes'})
        return valid

    def load_latest(self):
        rows = self._records()
        children = defaultdict(list)
        for row in rows:
            children[row['parent']].append(row)
        for values in children.values():
            if len(values) > 1 and sum(self._verify(row) for row in values) > 1:
                raise CheckpointConflict('Divergent checkpoint branches; do not run two writers')
        for row in sorted(rows, key=lambda r: r['step'], reverse=True):
            if row['file'] in self.invalid or not self._verify(row):
                continue
            state = torch.load(self.durable / row['file'], map_location='cpu', weights_only=True)
            if state['schema_version'] != 1 or state['step'] != row['step']:
                raise ValueError('Checkpoint payload and manifest disagree')
            self.parent, self.step = row['file'], row['step']
            return state
        return None

    def save(self, state):
        step = state['step']
        if state['schema_version'] != 1 or not isinstance(step, int) or step <= self.step:
            raise ValueError('Save only a later completed update boundary')
        rows = self._records()
        if rows and max(rows, key=lambda r: r['step'])['file'] != self.parent:
            raise CheckpointConflict('Another writer advanced this run, or resume was not loaded')
        with tempfile.NamedTemporaryFile(dir=self.local, suffix='.pt', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(state, stream)
            _flush(stream)
        checksum = sha256(temporary)
        name = f'step-{step:010d}-{checksum}.pt'
        staged = self.durable / (name + '.partial-' + uuid.uuid4().hex)
        try:
            with temporary.open('rb') as source, staged.open('xb') as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
                _flush(target)
            if sha256(staged) != checksum:
                raise ValueError('Checkpoint transfer read-back failed')
            destination = self.durable / name
            os.replace(staged, destination)
            row = {'schema_version': 1, 'file': name, 'bytes': destination.stat().st_size,
                   'sha256': checksum, 'identity_sha256': self.identity_hash,
                   'step': step, 'parent': self.parent}
            if not self._verify(row):
                raise ValueError('Final checkpoint read-back failed')
            write_json_verified(self.durable / (name[:-3] + '.complete.json'), row)
            self.parent, self.step = name, step
            return row
        finally:
            temporary.unlink(missing_ok=True)
