#!/usr/bin/env python3
"""Build the complete GPU study from original archives, without old model caches."""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = dict(cell_type=kind, metadata={}, source=source.strip() + "\n")
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--commit", required=True)
    args = p.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        raise ValueError("Pin a complete published implementation commit")
    previous = json.loads((ROOT / "notebooks/public_corpus_comparison.ipynb").read_text())

    def prior(i):
        return "".join(previous["cells"][i]["source"])

    cells = [
        cell(
            "markdown",
            """
# BEATs: positive-source diversity and full fine-tuning

Run all cells with a **GPU runtime**. This notebook starts from the downloaded
**222 original distributions (39.011 GB)**. It does **not** require a previous
trained model, feature archive or completed notebook. It also acquires the
6.26 GB FSD50K evaluation audio and bounded WSU/AuDroK source material.

The complete cohort has **30,238 canonical crops**. For each of two whole-source
holdouts (WSU and AuDroK), compare current versus broader positive training,
first with frozen BEATs and then with all **90.3 million** encoder parameters
trainable. Every neural arm visits every selected training crop for three
epochs. Model/threshold selection excludes the held source. This is development
validation on previously examined collections, not a new generalization test.
Publisher presence labels remain weak audibility labels.

Original archives are read only. Verified canonical audio, model parameters,
optimizer recovery checkpoints and results persist in the separate Drive project.
Training reads verified local binary copies for speed. Use **one active runtime
per project**; CPU and GPU studies have separate directories and recompute their
own matched frozen controls. Keep the same GPU type when resuming a neural arm.
The pipeline pauses at a batch checkpoint after a cumulative 44-hour budget.
It never replaces the study with a small sample. No automatic future execution
or notification is configured by this notebook.
""",
        ),
        cell(
            "code",
            f"""
ARCHIVES_DRIVE_PATH = '/content/drive/MyDrive/drone-detection-v2/corpus-intake-v3/source-archives'
PROJECT_DRIVE_PATH = '/content/drive/MyDrive/drone-beats-adaptation-v1'
REPO_URL = 'https://github.com/StoneDerek/drone-detection.git'
CODE_COMMIT = '{args.commit}'
DEVICE = 'cuda'
THREADS = 1
FEATURE_SHARDS = 1  # Bootstrap contract; training has a single writer.
if DEVICE != 'cuda':
    raise ValueError('Select a GPU runtime for this notebook; CPU is a separate study')
""",
        ),
        cell("code", prior(2)),
        cell("code", prior(3).replace("/content/drone-corpus-code", "/content/drone-beats-adaptation-code")),
        cell("code", prior(4)),
        cell(
            "markdown",
            """
## Verify the original distributions and prepare the full canonical corpus

The first run independently checks publisher digests for all 222 files. It uses
a local symlink view to add verification receipts without changing the originals.
Completed preparation is reused after a runtime replacement. Missing or
conflicting inputs fail explicitly. Archive verification and audio preparation
take time before training begins; no old embeddings are needed.
""",
        ),
        cell(
            "code",
            """
LOCAL = Path('/content/drone-beats-adaptation-inputs')
CANONICAL = PROJECT / 'canonical-inputs'
LOCAL.mkdir(parents=True, exist_ok=True)
CANONICAL.mkdir(parents=True, exist_ok=True)

def durable_link(link, target):
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise ValueError('Existing local input link points elsewhere: ' + str(link))
    elif link.exists():
        raise ValueError('Preserve the existing local input folder: ' + str(link))
    else:
        link.symlink_to(target, target_is_directory=True)

PREVIOUS = LOCAL / 'milestone-v1'
FIELD = LOCAL / 'confuser-study-v1'
durable_link(PREVIOUS / 'prepared', CANONICAL / 'original-prepared')
durable_link(FIELD / 'prepared', CANONICAL / 'field-prepared')
durable_link(FIELD / 'archives', CANONICAL / 'field-archives')
URBAN = LOCAL / 'UrbanSound8K.csv'
if not URBAN.exists():
    url = 'https://raw.githubusercontent.com/MOULIK-RAZDAN/AUDIO-CLASSIFICATION/main/UrbanSound8K.csv'
    with urllib.request.urlopen(url, timeout=60) as response:
        URBAN.write_bytes(response.read())
if hashlib.sha256(URBAN.read_bytes()).hexdigest() != 'b207f7a53f9e5eb02a5dd71d58fb8b66c88c24c0637b926dc72d9d400a6a2354':
    raise ValueError('Pinned UrbanSound metadata changed')
if not (PREVIOUS / 'prepared/preparation-complete.json').exists():
    VERIFIED_ARCHIVES = LOCAL / 'verified-source-archives'
    run_script('verify_adaptation_archives.py', '--archives', ARCHIVES, '--output', VERIFIED_ARCHIVES)
    run_script('prepare_corpus_study.py', '--archives', VERIFIED_ARCHIVES,
               '--output', PREVIOUS / 'prepared', '--urban-csv', URBAN)
    run_script('finalize_corpus_study.py', '--prepared', PREVIOUS / 'prepared', '--archives', VERIFIED_ARCHIVES)
if not (FIELD / 'prepared/preparation-complete.json').exists():
    run_script('acquire_confuser_study.py', '--work', FIELD, '--metadata-only')
    run_script('prepare_confuser_study.py', 'plan', '--work', FIELD, '--previous', PREVIOUS,
               '--urban-metadata', URBAN)
    run_script('acquire_confuser_study.py', '--work', FIELD)
    for stage in ['wsu', 'audrok', 'fsd', 'seal']:
        run_script('prepare_confuser_study.py', stage, '--work', FIELD)
completion = json.loads((FIELD / 'prepared/preparation-complete.json').read_text())
manifest = FIELD / 'prepared/manifest.jsonl'
if hashlib.sha256(manifest.read_bytes()).hexdigest() != completion['manifest_sha256']:
    raise ValueError('Canonical preparation receipt no longer matches its manifest')
print('Complete canonical corpus:', completion)
""",
        ),
        cell(
            "code",
            """
STAGED_CORPUS = LOCAL / 'training-inputs'
run_script('stage_adaptation_audio.py', '--manifest', manifest, '--output', STAGED_CORPUS)
DURABLE_ASSETS = PROJECT / 'model-assets/beats'
run_script('fetch_beats.py', '--output', DURABLE_ASSETS)
ASSETS = LOCAL / 'model-assets/beats'
shutil.copytree(DURABLE_ASSETS, ASSETS, dirs_exist_ok=True)
WORK = PROJECT / DEVICE / 'study'
REPORT = WORK / 'report'
run_script('run_beats_adaptation.py', 'plan', '--corpus', STAGED_CORPUS,
           '--work', WORK, '--report', REPORT)
shutil.copyfile(REPO / 'reports/beats-adaptation-v1/PROTOCOL.md', REPORT / 'PROTOCOL.md')
print('Whole-source folds and three-epoch policy frozen:', WORK / 'plan.json')
""",
        ),
        cell(
            "markdown",
            """
## Run the matched frozen controls and all four neural arms

The launcher first extracts native BEATs features for the complete cohort, then
fits four frozen controls, then trains four full-backbone models. Progress and
recovery state are saved continuously. A runtime restart resumes the optimizer
and exact next batch; completed stages are verified and reused. Partial feature
extraction is repeated if interrupted before its atomic matrix checkpoint.

The cell displays the current stage and recent log entries once per minute.
Interrupting the cell requests a cooperative pause; allow the child time to
finish its batch and write recovery state. Do not start another runtime writer.
A budget pause requires a deliberate decision before extending the compute
budget; rerunning the notebook does not silently grant more time.
""",
        ),
        cell(
            "code",
            """
import time

pause = WORK / 'pause-requested.json'
if pause.exists():
    raise RuntimeError('Study is paused. Preserve the checkpoint and review ' + str(pause))
command = [str(PYTHON), '-u', str(REPO / 'scripts/launch_beats_adaptation.py'),
           '--work', str(WORK), '--assets', str(ASSETS), '--python', str(PYTHON),
           '--device', DEVICE, '--threads', str(THREADS), '--max-wall-hours', '44']
supervisor_log = WORK / 'supervisor.log'
with supervisor_log.open('a') as log:
    process = subprocess.Popen(command, cwd=REPO, env=JOB_ENV, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    last_report = 0
    try:
        while process.poll() is None:
            if time.monotonic() - last_report >= 60:
                state_file = WORK / 'pipeline-state.json'
                if state_file.exists():
                    state = json.loads(state_file.read_text())
                    print('Pipeline:', state, flush=True)
                    if 'stage' in state:
                        stage_log = WORK / (state['stage'] + '.log')
                        if stage_log.exists():
                            with stage_log.open('rb') as source:
                                source.seek(max(0, stage_log.stat().st_size - 1600))
                                print(source.read().decode(errors='replace'), flush=True)
                last_report = time.monotonic()
            time.sleep(5)
    except KeyboardInterrupt:
        pause.write_text(json.dumps({'reason': 'notebook user interrupted; checkpoint requested'}))
        print('Checkpoint pause requested. Keep this runtime open until the study stops.', flush=True)
        raise
if process.returncode:
    raise RuntimeError('Study launcher failed; inspect ' + str(supervisor_log))
print(json.loads((WORK / 'pipeline-state.json').read_text()))
""",
        ),
        cell(
            "code",
            """
state = json.loads((WORK / 'pipeline-state.json').read_text())
if state['status'] != 'complete':
    raise RuntimeError('Full fine-tuning is not complete: ' + repr(state))
frozen = json.loads((WORK / 'frozen-controls.json').read_text())
adapted = json.loads((WORK / 'fine-tuned-controls.json').read_text())
table = []
for family, results in [('frozen', frozen), ('fine-tuned', adapted)]:
    for held_source, variants in results.items():
        for mix, result in variants.items():
            table.append({'encoder': family, 'held_source': held_source, 'positive_mix': mix,
                          'validation': result['validation']['by_source']})
print(json.dumps(table, indent=2))
print('Complete results and retained trained parameters:', WORK)
print('These are held-source development comparisons, using unchanged weak publisher labels.')
""",
        ),
    ]
    result = dict(
        cells=cells,
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python"),
            colab=dict(provenance=[], name="BEATs positive diversity and full fine-tuning"),
        ),
        nbformat=4,
        nbformat_minor=5,
    )
    for i, c in enumerate(cells):
        c["id"] = f"beats-adaptation-{i}"
        if c["cell_type"] == "code":
            compile(c["source"], f"cell{i}", "exec")
    (ROOT / "notebooks/beats_adaptation.ipynb").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
