#!/usr/bin/env python3
"""Build the Colab launcher for the broader original-source corpus intake."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source, name):
    result = {"cell_type": kind, "id": name, "metadata": {}, "source": source.strip() + "\n"}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def notebook():
    return {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"colab": {"name": "corpus_intake.ipynb", "provenance": []},
                     "kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": [
            cell("markdown", """# Broader original-source corpus intake

Use a **CPU runtime**. The default `core` profile adds **39.011 GB (36.332 GiB)**
of original releases: DroneNoise, Berne, Polish UAV/speech, NEAPTIDE, both TUT2017
releases, UrbanSound8K and DNC. Keep the completed AuDroK intake and its receipts.
The files include repeated microphones, calibration and processed measurements;
downloaded bytes do not establish independent hours, labels or field reliability.

Check available Drive space before running. `STORAGE_BUDGET_GIB` is your declared
allocation for this selection; it does not measure the account quota. The separate
folder below preserves earlier code pins and completed sources. Use one active
runtime for this folder. No packages, GPU allocation or model training are needed.
""", "introduction"),
            cell("code", """PROJECT_DRIVE_PATH = '/content/drive/MyDrive/drone-detection-v2/corpus-intake-v3'
INTAKE_PROFILE = 'core'  # Alternatives: 'field', 'background', 'metadata'.
STORAGE_BUDGET_GIB = 60.0  # Set from the available Drive allocation you have checked.
REPO_URL = 'https://github.com/StoneDerek/drone-detection.git'
FIRST_REF = 'main'
""", "settings"),
            cell("code", """import os
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
""", "mount-drive"),
            cell("code", (ROOT / "notebooks/bootstrap.py").read_text() + """

REPO = Path('/content/drone-corpus-intake-v3')
PIN = pin_checkout(PROJECT, REPO, REPO_URL, FIRST_REF)
print(json.dumps(PIN, indent=2))
""", "pinned-bootstrap"),
            cell("code", """import subprocess
import sys

COMMAND = [sys.executable, str(REPO / 'scripts/fetch_corpus_intake.py'),
           '--manifest', str(REPO / 'data/corpus-intake-v3.json'),
           '--profile', INTAKE_PROFILE,
           '--output', str(PROJECT / 'source-archives')]
subprocess.run(COMMAND + ['--plan-only'], cwd=REPO, check=True)
""", "show-selection"),
            cell("markdown", """Each completed file must match its pinned publisher checksum. The stored
file is read back to check that checksum and record SHA-256 before completion is
published. On a replacement runtime, completed files are rehashed and reused.
Interrupted files retain an **unverified partial** and an identity journal.
Rerunning resumes its exact byte offset where the server supports Range; a server
that ignores Range restarts that file. A changed source or corrupt prefix cannot
pass the final checksum. Preserve failed files for review rather than deleting
the intake folder.

This freezes acquisition state only. Keep native originals and per-source terms.
Do not extract the whole selection onto the notebook SSD. Recording/channel
parentage and interval labels will be prepared in a separate bounded cache step.
""", "recovery-notes"),
            cell("code", """subprocess.run(COMMAND + ['--budget-gib', str(STORAGE_BUDGET_GIB)],
               cwd=REPO, check=True)
print('Original distributions:', PROJECT / 'source-archives')
""", "download-sources"),
            cell("markdown", """Flush before ending a session. After replacing the runtime, rerun this
notebook with the same folder to verify completed-file persistence and partial
recovery. Mounted read-back is not independent proof of cloud sync; unexpected
termination can lose bytes not yet synchronized to Drive.

Source/label inspection and a grouped manifest follow acquisition. The 100-hour
target was an unvalidated estimate; training quantity will be assessed using
grouped learning curves and separate held-out evaluation exposure.
""", "next-step"),
            cell("code", """drive.flush_and_unmount()
print('Drive flushed and unmounted. Verify persistence on the replacement runtime.')
""", "flush-drive"),
        ]}


def main():
    (ROOT / "notebooks/corpus_intake.ipynb").write_text(json.dumps(notebook(), indent=1) + "\n")


if __name__ == "__main__":
    main()
