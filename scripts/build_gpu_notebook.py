#!/usr/bin/env python3
"""Build the reviewable Colab setup notebook; keep its bootstrap in one source."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source, name):
    result = {'cell_type': kind, 'id': name, 'metadata': {}, 'source': source.strip() + '\n'}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def notebook():
    return {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'colab': {'name': 'gpu_runtime_setup.ipynb', 'provenance': []},
                     'accelerator': 'GPU',
                     'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'language_info': {'name': 'python'}},
        'cells': [
            cell('markdown', '''# Drone detection: GPU runtime and recovery setup

This prepares the free Colab/Google Drive training host. **It does not train a
new drone detector.** The small numerical fixture tests checkpoint recovery;
it supplies no evidence about drone recognition.

1. Select **Runtime → Change runtime type → GPU** (a free GPU is not guaranteed).
2. Edit the project folder below if needed, then **Run all** and authorize Drive.
3. Check that the final report says `passed: true`.
4. Run the final flush cell. Select **Runtime → Disconnect and delete runtime**,
   reconnect with a GPU, and run all again. The second report should also say
   `replayed_checkpoint_from_earlier_invocation: true`.

Use only **one active runtime per project folder**. Colab may disconnect without
warning. Code and dependencies are recreated locally; checkpoints and the pinned
code revision live in Drive. Drive quota and source-data permissions must be
checked before any bulk data transfer. This notebook downloads no audio.
''', 'introduction'),
            cell('code', '''# Change this before the first run if you want a different Drive folder.
# A shared drive can use /content/drive/Shareddrives/<drive>/<folder>.
PROJECT_DRIVE_PATH = '/content/drive/MyDrive/drone-detection-v2'
REPO_URL = 'https://github.com/StoneDerek/drone-detection.git'
FIRST_REF = 'main'  # Used only when bootstrap.json does not yet exist.
''', 'settings'),
            cell('code', '''import hashlib
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from google.colab import drive

# Stop before installing the CUDA environment if a GPU was not allocated.
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                '--format=csv'], check=True)
drive.mount('/content/drive')
MOUNT = Path('/content/drive')
if not os.path.ismount(MOUNT):
    raise RuntimeError('Google Drive is not mounted; refusing an ephemeral checkpoint folder')
PROJECT = Path(PROJECT_DRIVE_PATH).resolve()
if not PROJECT.is_relative_to(MOUNT.resolve()) or PROJECT == MOUNT.resolve():
    raise ValueError('Choose a project folder inside the mounted Google Drive')
PROJECT.mkdir(parents=True, exist_ok=True)
payload = os.urandom(1024 * 1024)
probe = PROJECT / ('write-check-' + uuid.uuid4().hex + '.bin')
try:
    probe.write_bytes(payload)
    assert hashlib.sha256(probe.read_bytes()).digest() == hashlib.sha256(payload).digest()
finally:
    probe.unlink(missing_ok=True)
print('Drive mount write/read check passed. This does not measure quota or prove cloud sync.')
print('Local disk:', shutil.disk_usage('/content'))
''', 'mount-and-device'),
            cell('code', (ROOT / 'notebooks/bootstrap.py').read_text() + '''

REPO = Path('/content/drone-detection')
PIN = pin_checkout(PROJECT, REPO, REPO_URL, FIRST_REF)
print(json.dumps(PIN, indent=2))
''', 'pinned-bootstrap'),
            cell('markdown', '''The first run resolves `main` once and saves `bootstrap.json` in Drive.
Later runs fetch that exact commit and verify its environment lock. Updates to
`main` will not silently change an interrupted job. To start a newer study, use
a new project folder and retain the old checkpoints.

The CUDA environment below is separate from Colab's notebook kernel. It pins
PyTorch 2.7.1 with CUDA 12.6 and NumPy; setup fails if the allocated GPU/driver
cannot run it. All training commands run in child processes using this environment.
''', 'environment-notes'),
            cell('code', '''import sys

UV_DIR = Path('/content/drone-bootstrap')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade',
                '--target', str(UV_DIR), 'uv==0.13.5'], check=True)
UV_ENV = dict(os.environ, PYTHONPATH=str(UV_DIR),
              UV_CACHE_DIR='/content/drone-uv-cache',
              UV_PYTHON_INSTALL_DIR='/content/drone-python')
subprocess.run([sys.executable, '-m', 'uv', 'sync', '--project',
                str(REPO / 'notebooks/runtime'), '--frozen', '--python', '3.12',
                '--extra', 'gpu'], env=UV_ENV, check=True)
PYTHON = REPO / 'notebooks/runtime/.venv/bin/python'
JOB_ENV = dict(os.environ, PYTHONPATH=str(REPO / 'src'),
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
PROBE_SCRIPT = REPO / 'scripts/notebook_recovery_probe.py'
subprocess.run([str(PYTHON), str(PROBE_SCRIPT), 'preflight', '--device', 'cuda'],
               cwd=REPO, env=JOB_ENV, check=True)
''', 'locked-environment'),
            cell('markdown', '''The recovery check runs an uninterrupted control, then kills a separate
process after update 5. Only update 4 has a complete checkpoint. A fresh process
must restore it and reach update 12 with matching model/buffers, optimizer,
scheduler, AMP scaler, random state, data order and selection history.

On later notebook invocations it replays the original persisted update-4
checkpoint in a new directory. It does not merely read a finished result.
A different GPU can change numerical results; a mismatch stops the check rather
than passing silently. This fixture uses one data worker (`num_workers=0`);
a future loader needs its own recovery check.
''', 'recovery-notes'),
            cell('code', '''DURABLE = PROJECT / 'runtime-checks/cuda-v1'
LOCAL = Path('/content/drone-recovery-scratch')
subprocess.run([str(PYTHON), str(PROBE_SCRIPT), 'verify', '--device', 'cuda',
                '--durable', str(DURABLE), '--local', str(LOCAL)],
               cwd=REPO, env=JOB_ENV, check=True)
report = json.loads((DURABLE / 'probe-report.json').read_text())
assert report['passed']
print('Saved report:', DURABLE / 'probe-report.json')
print('Replayed an earlier invocation:', report['replayed_checkpoint_from_earlier_invocation'])
''', 'verify-recovery'),
            cell('markdown', '''**Before disconnecting:** run the next cell to ask Colab to flush and
unmount Drive. Then replace the runtime and repeat the notebook. A mounted
read-back alone does not independently prove that bytes reached cloud storage.
The repeat after runtime deletion is the persistence check.

This is runtime preparation. Next come the source/label/duplicate inventory and
coverage gate in [the research plan](https://github.com/StoneDerek/drone-detection/blob/main/docs/research-plan-v2.md),
then an actual audio throughput pilot and integration of the selected CNN with
this recovery mechanism. The old development recordings are not a fresh test set.
''', 'next-step'),
            cell('code', '''drive.flush_and_unmount()
print('Drive flush/unmount completed. You can now replace the runtime and rerun.')
''', 'flush-drive'),
        ]}


def main():
    path = ROOT / 'notebooks/gpu_runtime_setup.ipynb'
    path.write_text(json.dumps(notebook(), indent=1) + '\n')


if __name__ == '__main__':
    main()
