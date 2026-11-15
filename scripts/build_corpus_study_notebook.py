#!/usr/bin/env python3
"""Build a full-corpus Colab handoff pinned to an already published code commit."""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source, identifier):
    result = {"cell_type": kind, "id": identifier, "metadata": {}, "source": source.strip() + "\n"}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def notebook(commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Pin a full published commit SHA, not a branch or placeholder")
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"name": "public_corpus_comparison.ipynb", "provenance": []},
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            cell(
                "markdown",
                """# Public-corpus drone classification comparison

This runs the actual eight-release corpus study: spectral, frozen CLAP, frozen
BEATs, and a level/duration control. Every eligible file contributes a fixed
central crop of at most three seconds. There is no small-example fallback.

1. Select **Runtime → Change runtime type → GPU**, keeping `DEVICE = 'cuda'`.
2. Confirm the incoming archive-folder path and use a separate results folder.
3. Run all cells and authorize your own Drive mount.
4. Inspect the accepted cohort counts and the final per-source results.

The input archives are read in place. Prepared audio and inference run on local
runtime storage. Numerical feature checkpoints and model/results files persist
to a separate Drive project. Completed feature families resume after a runtime
replacement. Keep one active notebook per results project.

This is weak-label crop classification. The recorded-file hours are not all used
as model input, and the results do not establish continuous-detector event recall
or false alarms per hour. Published data terms remain per release.
""",
                "introduction",
            ),
            cell(
                "code",
                f"""ARCHIVES_DRIVE_PATH = '/content/drive/MyDrive/drone-detection-v2/corpus-intake-v3/source-archives'
PROJECT_DRIVE_PATH = '/content/drive/MyDrive/drone-corpus-study-v1-20261115'
REPO_URL = 'https://github.com/StoneDerek/drone-detection.git'
CODE_COMMIT = '{commit}'
DEVICE = 'cuda'  # Explicitly choose 'cpu' for a longer CPU-only run.
FEATURE_SHARDS = 1  # Use 2 for two concurrent CPU extraction workers.
""",
                "settings",
            ),
            cell(
                "code",
                """import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from google.colab import drive

if DEVICE not in {'cpu', 'cuda'} or FEATURE_SHARDS not in {1, 2, 3, 4}:
    raise ValueError('Choose an explicit supported device and worker count')
if DEVICE == 'cuda':
    subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'], check=True)
drive.mount('/content/drive')
MOUNT = Path('/content/drive').resolve()
if not os.path.ismount(MOUNT):
    raise RuntimeError('Drive must be mounted for durable results')
PROJECT = Path(PROJECT_DRIVE_PATH).resolve()
ARCHIVES = Path(ARCHIVES_DRIVE_PATH).resolve()
if not PROJECT.is_relative_to(MOUNT) or PROJECT == MOUNT or not ARCHIVES.is_relative_to(MOUNT):
    raise ValueError('Choose the expected mounted Drive paths')
if PROJECT == ARCHIVES or PROJECT.is_relative_to(ARCHIVES) or ARCHIVES.is_relative_to(PROJECT):
    raise ValueError('Keep the result project separate from the input archives')
PROJECT.mkdir(parents=True, exist_ok=True)
if (PROJECT / 'bootstrap.json').exists() and json.loads((PROJECT / 'bootstrap.json').read_text()).get('revision') != CODE_COMMIT:
    raise ValueError('This project has another code pin; preserve it and use a new project folder')
print('Input archives:', ARCHIVES)
print('Durable results:', PROJECT)
print('Local runtime disk:', shutil.disk_usage('/content'))
""",
                "mount-and-paths",
            ),
            cell(
                "code",
                (ROOT / "notebooks/bootstrap.py").read_text()
                + """

REPO = Path('/content/drone-corpus-code')
PIN = pin_checkout(PROJECT, REPO, REPO_URL, CODE_COMMIT)
assert PIN['revision'] == CODE_COMMIT
print('Exact implementation:', PIN['revision'])
print('Runtime lock SHA-256:', PIN['runtime_lock_sha256'])
""",
                "exact-bootstrap",
            ),
            cell(
                "code",
                """UV_DIR = Path('/content/drone-corpus-bootstrap')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade', '--target', str(UV_DIR), 'uv==0.13.5'], check=True)
UV_ENV = dict(os.environ, PYTHONPATH=str(UV_DIR), UV_CACHE_DIR='/content/drone-corpus-uv-cache', UV_PYTHON_INSTALL_DIR='/content/drone-corpus-python')
subprocess.run([sys.executable, '-m', 'uv', 'sync', '--project', str(REPO / 'notebooks/runtime'), '--frozen', '--python', '3.12',
                '--extra', 'gpu' if DEVICE == 'cuda' else 'cpu', '--extra', 'probe'], env=UV_ENV, check=True)
PYTHON = REPO / 'notebooks/runtime/.venv/bin/python'
JOB_ENV = dict(os.environ, PYTHONPATH=str(REPO / 'src'), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
check = "import torch; print('Torch:',torch.__version__); "
check += "assert torch.cuda.is_available(); print('GPU:',torch.cuda.get_device_name(0))" if DEVICE == 'cuda' else "print('Explicit CPU execution; full corpus will still be processed')"
subprocess.run([str(PYTHON), '-c', check], cwd=REPO, env=JOB_ENV, check=True)

def run_script(name, *arguments):
    command = [str(PYTHON), str(REPO / 'scripts' / name), *[str(value) for value in arguments]]
    subprocess.run(command, cwd=REPO, env=JOB_ENV, check=True)
""",
                "locked-runtime",
            ),
            cell(
                "code",
                """STUDY = Path('/content/drone-corpus-study-' + DEVICE)
identity = {'project': str(PROJECT), 'archives': str(ARCHIVES), 'code': CODE_COMMIT, 'device': DEVICE}
identity_path = STUDY / 'local-study.json'
if STUDY.exists() and any(STUDY.iterdir()) and (not identity_path.exists() or json.loads(identity_path.read_text()) != identity):
    raise ValueError('Preserve the existing local study before starting an unrelated one')
STUDY.mkdir(parents=True, exist_ok=True)
identity_path.write_text(json.dumps(identity, sort_keys=True, indent=2))
MODEL_RESULTS = PROJECT / DEVICE / 'models'
MODEL_RESULTS.mkdir(parents=True, exist_ok=True)
for name, target in [('archives', ARCHIVES), ('models', MODEL_RESULTS)]:
    link = STUDY / name
    if not link.exists():
        link.symlink_to(target, target_is_directory=True)
    if link.resolve() != target:
        raise ValueError('Unexpected local study link: ' + name)
intake = json.loads((REPO / 'data/corpus-intake-v3.json').read_text())
missing = []
for item in intake['items']:
    source = ARCHIVES / item['source_id'] / item['filename']
    complete = source.with_name(source.name + '.complete.json')
    if not source.exists() or not complete.exists():
        missing.append(item['source_id'] + '/' + item['filename'])
        continue
    receipt = json.loads(complete.read_text())
    if source.stat().st_size != item['bytes'] or receipt.get('md5') != item['md5'] or not receipt.get('publisher_checksum_verified'):
        raise ValueError('Unverified source: ' + str(source))
if missing:
    raise RuntimeError('Finish the full intake before this study; missing distributions: ' + repr(missing))
print('Complete source inventory:', len(intake['items']), 'files;', sum(i['bytes'] for i in intake['items']), 'bytes')

URBAN = STUDY / 'urbansound.csv'
if not URBAN.exists():
    url = 'https://raw.githubusercontent.com/MOULIK-RAZDAN/AUDIO-CLASSIFICATION/main/UrbanSound8K.csv'
    with urllib.request.urlopen(url, timeout=60) as response:
        URBAN.write_bytes(response.read())
assert hashlib.sha256(URBAN.read_bytes()).hexdigest() == 'b207f7a53f9e5eb02a5dd71d58fb8b66c88c24c0637b926dc72d9d400a6a2354'
PREPARED = STUDY / 'prepared'
run_script('prepare_corpus_study.py', '--archives', ARCHIVES, '--output', PREPARED, '--urban-csv', URBAN)
run_script('finalize_corpus_study.py', '--prepared', PREPARED, '--archives', ARCHIVES)
cohort = json.loads((PREPARED / 'preparation-complete.json').read_text())
print(json.dumps(cohort, indent=2))
""",
                "full-corpus-preparation",
            ),
            cell(
                "code",
                """ASSETS = Path('/content/drone-corpus-assets')
for family in ['beats', 'clap']:
    durable = PROJECT / 'model-assets' / family
    run_script('fetch_' + family + '.py', '--output', durable)
    local = ASSETS / family
    local.mkdir(parents=True, exist_ok=True)
    for item in durable.iterdir():
        if item.is_file() and not item.name.endswith('.partial'):
            shutil.copy2(item, local / item.name)
print('Pinned model assets staged locally; the adapters recheck their hashes before loading')
""",
                "model-assets",
            ),
            cell(
                "code",
                """FEATURES = STUDY / 'features'
CHECKPOINTS = PROJECT / DEVICE / 'feature-checkpoints'
CHECKPOINTS.mkdir(parents=True, exist_ok=True)

def extract(family, splits):
    output = FEATURES / family
    checkpoint = CHECKPOINTS / (family + '.tar.gz')
    run_script('checkpoint_corpus_features.py', 'restore', '--directory', output, '--checkpoint', checkpoint)
    if FEATURE_SHARDS == 1:
        command = ['--prepared', PREPARED, '--output', output, '--family', family, '--threads', '1', '--device', DEVICE, '--splits', *splits]
        if family != 'spectral':
            command += ['--assets', ASSETS / family]
        run_script('extract_corpus_study_features.py', *command)
    else:
        command = ['--prepared', PREPARED, '--output', output, '--workspace', STUDY / 'feature-shards', '--python', PYTHON,
                   '--family', family, '--shards', str(FEATURE_SHARDS), '--device', DEVICE, '--splits', *splits]
        if family != 'spectral':
            command += ['--assets', ASSETS / family]
        run_script('shard_corpus_features.py', *command)
    run_script('checkpoint_corpus_features.py', 'pack', '--directory', output, '--checkpoint', checkpoint)
    print(family, 'requested partitions:', splits, 'completed cached records:', len(json.loads((output / 'index.json').read_text())))

for family in ['spectral', 'clap', 'beats']:
    extract(family, ['train', 'development'])
""",
                "training-development-features",
            ),
            cell(
                "code",
                """run_script('run_corpus_study.py', 'freeze', '--prepared', PREPARED, '--features', FEATURES, '--output', STUDY / 'models')
frozen = json.loads((MODEL_RESULTS / 'frozen-models.json').read_text())
assert frozen['plan']['manifest_sha256'] == hashlib.sha256((PREPARED / 'manifest.jsonl').read_bytes()).hexdigest()
print('Fitted training crops:', frozen['plan']['training_clips'])
print('Development crops:', frozen['plan']['development_clips'])
print('Every head and global cutoff is frozen and saved on Drive before final evaluation')
""",
                "freeze-all-decisions",
            ),
            cell(
                "code",
                """if not (MODEL_RESULTS / 'comparison.json').exists():
    for family in ['spectral', 'clap', 'beats']:
        extract(family, ['final'])
    run_script('run_corpus_study.py', 'evaluate', '--prepared', PREPARED, '--features', FEATURES, '--output', STUDY / 'models')
else:
    print('Keeping the saved final result; no new scoring or independent-test claim')
REPORT = PROJECT / DEVICE / 'report'
run_script('report_corpus_study.py', '--study', STUDY, '--output', REPORT)
from IPython.display import Markdown, display
display(Markdown((REPORT / 'RESULTS.md').read_text()))
print('Durable report:', REPORT)
print('A final cohort used for subsequent tuning is development material thereafter')
""",
                "final-comparison-and-report",
            ),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    path = ROOT / "notebooks/public_corpus_comparison.ipynb"
    path.write_text(json.dumps(notebook(args.code_commit), indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    main()
