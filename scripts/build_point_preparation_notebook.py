#!/usr/bin/env python3
"""Generate the versioned CPU point-file preparation launcher."""

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
        'metadata': {'colab': {'name': 'prepare_point_recordings.ipynb', 'provenance': []},
                     'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'language_info': {'name': 'python'}},
        'cells': [
            cell('markdown', '''# Prepare the individual-microphone recordings — CPU runtime

This reads the completed AuDroK intake and copies **58 point-microphone WAVs
(1.867 GB; about 2.94 file-hours)** unchanged to a new Drive folder. Three array
means per measurement are excluded. Select a **CPU runtime**.

Each source archive is checked against the frozen intake SHA, staged on local
SSD, and each selected member receives a complete CRC, finite-audio and native
format/frame check. No resampling, normalization or window generation occurs.
This preserves raw input for later study; it does not establish audible event
labels, independent sessions, negative exposure or model performance.

Check the paths and available additional Drive allocation below, then Run all.
Use one active runtime per preparation folder. Completed files resume after an
interruption; original archives remain untouched. Working copies need about 4 GB
of local SSD and are removed after their selected members finish.
''', 'introduction'),
            cell('code', '''SOURCE_DRIVE_PATH = '/content/drive/MyDrive/drone-detection-v2/intake-v1/source-archives'
PROJECT_DRIVE_PATH = '/content/drive/MyDrive/drone-detection-v2/point-preparation-v1'
ADDITIONAL_DRIVE_BUDGET_GIB = 3.0  # Available new allocation you have checked.
REPO_URL = 'https://github.com/StoneDerek/drone-detection.git'
FIRST_REF = 'main'
''', 'settings'),
            cell('code', '''import os
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')
MOUNT = Path('/content/drive')
if not os.path.ismount(MOUNT):
    raise RuntimeError('Drive is not mounted')
SOURCES, PROJECT = Path(SOURCE_DRIVE_PATH).resolve(), Path(PROJECT_DRIVE_PATH).resolve()
if any(not path.is_relative_to(MOUNT.resolve()) or path == MOUNT.resolve() for path in [SOURCES, PROJECT]):
    raise ValueError('Choose project folders inside mounted Drive')
if not SOURCES.is_dir():
    raise ValueError('Source folder is missing; check SOURCE_DRIVE_PATH')
if PROJECT.is_relative_to(SOURCES) or SOURCES.is_relative_to(PROJECT):
    raise ValueError('Keep preparation output separate from original source storage')
PROJECT.mkdir(parents=True, exist_ok=True)
''', 'mount-drive'),
            cell('code', (ROOT / 'notebooks/bootstrap.py').read_text() + '''

REPO = Path('/content/drone-point-preparation')
PIN = pin_checkout(PROJECT, REPO, REPO_URL, FIRST_REF)
print(json.dumps(PIN, indent=2))
''', 'pinned-bootstrap'),
            cell('code', '''import subprocess
import sys

UV_DIR = Path('/content/drone-point-bootstrap')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade',
                '--target', str(UV_DIR), 'uv==0.13.5'], check=True)
UV_ENV = dict(os.environ, PYTHONPATH=str(UV_DIR), UV_CACHE_DIR='/content/drone-point-uv-cache',
              UV_PYTHON_INSTALL_DIR='/content/drone-point-python')
subprocess.run([sys.executable, '-m', 'uv', 'sync', '--project', str(REPO),
                '--frozen', '--python', '3.12'], env=UV_ENV, check=True)
PYTHON = REPO / '.venv/bin/python'
JOB_ENV = dict(os.environ, PYTHONPATH=str(REPO / 'src'))
''', 'locked-environment'),
            cell('code', '''subprocess.run([str(PYTHON), str(REPO / 'scripts/prepare_point_cache.py'),
                '--sources', str(SOURCES), '--output', str(PROJECT / 'prepared'),
                '--cache', '/content/drone-point-working-sources',
                '--budget-gib', str(ADDITIONAL_DRIVE_BUDGET_GIB)],
               cwd=REPO, env=JOB_ENV, check=True)
summary = json.loads((PROJECT / 'prepared/summary.json').read_text())
assert summary['completed_point_files'] == 58 and summary['native_bytes_unchanged']
print(json.dumps(summary, indent=2))
''', 'prepare-recordings'),
            cell('markdown', '''Run the flush cell before a planned disconnect. After replacing the runtime,
rerun this same notebook: it reuses its saved code/environment pin and verifies
completed point files without staging their original archives again. Missing
files are reconstructed from the locked sources. Changed/corrupt files or run
identities stop execution for review.

The output contains the original native WAV bytes, per-file receipts, quality
measurements and an unassigned recording index. Publisher recording labels stay
separate from audible event intervals and reviewed negatives. The study still
needs additional independent campaigns and matched negative exposure.
''', 'next-step'),
            cell('code', '''drive.flush_and_unmount()
print('Drive flushed and unmounted. Prepared files can be verified on a replacement runtime.')
''', 'flush-drive'),
        ]}


def main():
    (ROOT / 'notebooks/prepare_point_recordings.ipynb').write_text(json.dumps(notebook(), indent=1) + '\n')


if __name__ == '__main__':
    main()
