#!/usr/bin/env python3
"""Build the CPU-only public source-intake launcher for Colab."""

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
        'metadata': {'colab': {'name': 'source_intake.ipynb', 'provenance': []},
                     'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'language_info': {'name': 'python'}},
        'cells': [
            cell('markdown', '''# Public source intake — CPU runtime

This downloads **6.890 GB** with the default `outdoor` profile: two original
AuDroK outdoor archives, the small dual-drone archive and catalog, and one
RealMAN background microphone file used to verify selective extraction.
No GPU or model training is involved. Select a **CPU runtime**.

Confirm enough available Drive allocation before running. `STORAGE_BUDGET_GIB`
is your declared project allocation; it does not measure the account's quota.
The `pilot` profile downloads only 206 MB. Both profiles preserve native sources.

This is acquisition for inspection. Repeated microphones are not independent
capture time, publisher labels still need review, and this intake does not pass
the full corpus diversity/quality gate. Keep one active runtime per intake folder.
''', 'introduction'),
            cell('code', '''PROJECT_DRIVE_PATH = '/content/drive/MyDrive/drone-detection-v2/intake-v1'
INTAKE_PROFILE = 'outdoor'  # Or 'pilot' for the 206 MB verified sample.
STORAGE_BUDGET_GIB = 20.0  # Set to the available allocation you have checked.
REPO_URL = 'https://github.com/StoneDerek/drone-detection.git'
FIRST_REF = 'main'
''', 'settings'),
            cell('code', '''import os
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')
MOUNT = Path('/content/drive')
if not os.path.ismount(MOUNT):
    raise RuntimeError('Drive is not mounted; refusing an ephemeral intake folder')
PROJECT = Path(PROJECT_DRIVE_PATH).resolve()
if PROJECT == MOUNT.resolve() or not PROJECT.is_relative_to(MOUNT.resolve()):
    raise ValueError('Choose a project folder inside mounted Drive')
PROJECT.mkdir(parents=True, exist_ok=True)
''', 'mount-drive'),
            cell('code', (ROOT / 'notebooks/bootstrap.py').read_text() + '''

REPO = Path('/content/drone-source-intake')
PIN = pin_checkout(PROJECT, REPO, REPO_URL, FIRST_REF)
print(json.dumps(PIN, indent=2))
''', 'pinned-bootstrap'),
            cell('markdown', '''This intake has its own code pin, separate from the completed GPU recovery
probe. Replacement runtimes reuse its exact commit. No packages are installed;
the downloader uses Python's standard library.

The known pilot files must match pinned SHA-256 values. The two large outdoor
archives have no publisher checksum; their first full transfer checks the
advertised byte count, hashes the bytes, verifies storage read-back and freezes
the observed hash in a receipt. This is not publisher checksum verification.

AuDroK's server ignores HTTP ranges. If a runtime dies during an archive, that
archive restarts; completed archives are rehashed and reused. Partial files are
not accepted as complete. A rerun rejects changed/corrupt sources and receipts.
''', 'verification-notes'),
            cell('code', '''import subprocess
import sys

subprocess.run([sys.executable, str(REPO / 'scripts/fetch_source_intake.py'),
                '--manifest', str(REPO / 'data/source-intake-v2.json'),
                '--profile', INTAKE_PROFILE,
                '--output', str(PROJECT / 'source-archives'),
                '--budget-gib', str(STORAGE_BUDGET_GIB)], cwd=REPO, check=True)
print('Source directory:', PROJECT / 'source-archives')
''', 'download-sources'),
            cell('markdown', '''Run the flush cell, replace the runtime, and repeat this notebook to check
that completed sources persist and are reused. Mounted read-back alone does not
independently prove cloud sync. Keep source archives and receipts together.

Next comes archive/member inspection, microphone/recording parentage, native
precision and interval validation, followed by a frozen data manifest. No
train/test assignment or new detector result is produced by this notebook.
''', 'next-step'),
            cell('code', '''drive.flush_and_unmount()
print('Drive flushed and unmounted. Completed sources can be verified on the next runtime.')
''', 'flush-drive'),
        ]}


def main():
    path = ROOT / 'notebooks/source_intake.ipynb'
    path.write_text(json.dumps(notebook(), indent=1) + '\n')


if __name__ == '__main__':
    main()
