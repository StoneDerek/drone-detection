#!/usr/bin/env python3
"""Build the Colab smoke handoff, pinned to its published implementation commit."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_COMMIT = 'd49fa175f2f739c3ca0ba2a1a049a5d71fa5bf5e'


def cell(kind, source, name):
    result = {'cell_type': kind, 'id': name, 'metadata': {}, 'source': source.strip() + '\n'}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def notebook():
    return {
        'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {
            'colab': {'name': 'frozen_audio_comparison.ipynb', 'provenance': []},
            'accelerator': 'GPU',
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
        },
        'cells': [
            cell('markdown', '''# Frozen BEATs and CLAP: Colab software check

This runs the published comparison code on five retained, already-consumed
browser excerpts: three training and two development recordings. It verifies
model loading, the complete comparison, saved numeric heads and cache reuse.
The examples do not establish generalization or useful detector accuracy.

1. Open a separate Colab session and select **Runtime → Change runtime type → GPU**.
2. Keep `DEVICE = 'cuda'`, or explicitly choose `'cpu'` for a CPU-only check.
3. Run all cells and authorize Google Drive when prompted.
4. Check that both encoders finish and every completed rerun reports 5 cache hits.
5. The last cell flushes and unmounts this session's Drive. Delete/replace the
   runtime and rerun to check recovery from the same Drive-backed assets/results.

The notebook uses a new Drive project folder and leaves the bulk audio-download
folder and its bootstrap pin untouched. It downloads approximately 0.98 GB of
pretrained model assets; it downloads no new audio. Keep one active writer per
smoke project. Actual GPU execution and Drive persistence must be checked here;
the preceding local CPU verification cannot establish either.
''', 'introduction'),
            cell('code', f'''PROJECT_DRIVE_PATH = '/content/drive/MyDrive/drone-detection-frozen-audio-20261115'
REPO_URL = 'https://github.com/StoneDerek/drone-detection.git'
CODE_COMMIT = '{CODE_COMMIT}'
DEVICE = 'cuda'  # Explicitly set 'cpu' to use a standard CPU-only runtime.
ENCODERS = ('beats', 'clap')
''', 'settings'),
            cell('code', '''import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from google.colab import drive

if DEVICE not in {'cpu', 'cuda'}:
    raise ValueError("Choose DEVICE='cpu' or DEVICE='cuda'")
if DEVICE == 'cuda':
    subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                    '--format=csv'], check=True)
if not ENCODERS or set(ENCODERS) - {'beats', 'clap'}:
    raise ValueError('Choose one or both declared encoders')
drive.mount('/content/drive')
MOUNT = Path('/content/drive')
if not os.path.ismount(MOUNT):
    raise RuntimeError('Drive is not mounted; refusing ephemeral result storage')
PROJECT = Path(PROJECT_DRIVE_PATH).resolve()
if not PROJECT.is_relative_to(MOUNT.resolve()) or PROJECT == MOUNT.resolve():
    raise ValueError('Choose a project folder inside mounted Drive')
PROJECT.mkdir(parents=True, exist_ok=True)
existing = PROJECT / 'bootstrap.json'
if existing.exists() and json.loads(existing.read_text()).get('revision') != CODE_COMMIT:
    raise ValueError('This project already has another code pin; use a new folder')
print('Smoke project:', PROJECT)
print('Available runtime disk:', shutil.disk_usage('/content'))
''', 'mount-and-device'),
            cell('code', (ROOT / 'notebooks/bootstrap.py').read_text() + '''

REPO = Path('/content/drone-frozen-code')
PIN = pin_checkout(PROJECT, REPO, REPO_URL, CODE_COMMIT)
assert PIN['revision'] == CODE_COMMIT
print('Exact code revision:', PIN['revision'])
print('Pinned runtime lock:', PIN['runtime_lock_sha256'])
''', 'pinned-bootstrap'),
            cell('code', '''UV_DIR = Path('/content/drone-frozen-bootstrap')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade',
                '--target', str(UV_DIR), 'uv==0.13.5'], check=True)
UV_ENV = dict(os.environ, PYTHONPATH=str(UV_DIR),
              UV_CACHE_DIR='/content/drone-frozen-uv-cache',
              UV_PYTHON_INSTALL_DIR='/content/drone-frozen-python')
subprocess.run([sys.executable, '-m', 'uv', 'sync', '--project',
                str(REPO / 'notebooks/runtime'), '--frozen', '--python', '3.12',
                '--extra', 'gpu' if DEVICE == 'cuda' else 'cpu',
                '--extra', 'probe', '--extra', 'test'], env=UV_ENV, check=True)
PYTHON = REPO / 'notebooks/runtime/.venv/bin/python'
JOB_ENV = dict(os.environ, PYTHONPATH=str(REPO / 'src'),
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
check = "import torch; print('Torch:',torch.__version__); "
if DEVICE == 'cuda':
    check += "assert torch.cuda.is_available(); print('GPU:',torch.cuda.get_device_name(0))"
else:
    check += "print('Explicit CPU execution selected')"
subprocess.run([str(PYTHON), '-c', check], cwd=REPO, env=JOB_ENV, check=True)
''', 'locked-environment'),
            cell('code', '''SMOKE_DATA = PROJECT / 'smoke-data'
subprocess.run([str(PYTHON), str(REPO / 'scripts/prepare_frozen_probe_smoke.py'),
                '--output', str(SMOKE_DATA)], cwd=REPO, env=JOB_ENV, check=True)
MANIFEST = SMOKE_DATA / 'manifest.jsonl'
receipt = json.loads((SMOKE_DATA / 'manifest-receipt.json').read_text())
assert receipt['recordings'] == 5 and receipt['verified_temporal_annotations'] == 0
print(receipt['scope'])
print('Training:', len(receipt['train']), 'Development:', len(receipt['development']))
''', 'prepare-retained-audio'),
            cell('code', '''LOCAL_ASSETS = Path('/content/drone-frozen-assets') / CODE_COMMIT
for family in ENCODERS:
    durable = PROJECT / 'model-assets' / family
    subprocess.run([str(PYTHON), str(REPO / f'scripts/fetch_{family}.py'),
                    '--output', str(durable)], cwd=REPO, env=JOB_ENV, check=True)
    local = LOCAL_ASSETS / family
    local.mkdir(parents=True, exist_ok=True)
    for member in durable.iterdir():
        if member.is_file() and not member.name.endswith('.partial'):
            shutil.copy2(member, local / member.name)
    # The runner verifies staged asset hashes before loading any model code/weights.
    print('Verified durable assets staged for', family)
''', 'verified-model-assets'),
            cell('code', '''def run_comparison(family):
    output = PROJECT / 'smoke-results' / DEVICE / family
    cache = PROJECT / 'feature-cache' / DEVICE / family
    command = [str(PYTHON), str(REPO / 'scripts/run_frozen_probe.py'),
               '--manifest', str(MANIFEST), '--encoder-family', family,
               '--encoder', str(LOCAL_ASSETS / family), '--device', DEVICE,
               '--threads', '2', '--output', str(output), '--cache', str(cache),
               '--reference-bundle', str(REPO / 'artifacts/real-v2/baseline')]
    subprocess.run(command, cwd=REPO, env=JOB_ENV, check=True)
    report = json.loads((output / 'comparison.json').read_text())
    assert report['training_recordings'] == 3 and report['development_recordings'] == 2
    assert report['feature_recordings'] == 5 and not report['field_reliability_claim']
    return report, output

for family in ENCODERS:
    first, output = run_comparison(family)
    repeated, _ = run_comparison(family)  # Fresh child process; completed features must be reused.
    assert repeated['cache_hits'] == 5
    assert repeated['model_id'] == first['model_id']
    assert repeated['predictions'] == first['predictions']
    print(f"{family}: saved head/predictions reproduced; all 5 cache entries reused")
    print('Results:', output)
print('Colab smoke PASSED. No generalization claim follows from these five examples.')
''', 'run-and-verify-comparisons'),
            cell('markdown', '''Timestamped diagnostic scores, heads, comparison reports and caches are
saved beneath this separate Drive project. The result counts are not a new
accuracy benchmark. The larger corpus needs native-file checks, label mappings,
recording-group splits and a reserved evaluation set before the next study.

The complete rerun tests fresh-process cache/head reuse. Actual runtime deletion
and a later rerun are needed to test Drive persistence. A GPU can change after
runtime replacement; cached embeddings retain their original numeric values.
The earlier CPU forced-interruption test is separate from this Colab check.

The final cell flushes/unmounts this session's Drive. It does not alter the bulk
intake project's bootstrap file or source archives.
''', 'recovery-and-limits'),
            cell('code', '''drive.flush_and_unmount()
print('Drive flushed and unmounted. Replace the runtime and rerun this notebook to check persistence.')
''', 'flush-drive'),
        ],
    }


def main():
    path = ROOT / 'notebooks/frozen_audio_comparison.ipynb'
    path.write_text(json.dumps(notebook(), indent=1) + '\n')


if __name__ == '__main__':
    main()
