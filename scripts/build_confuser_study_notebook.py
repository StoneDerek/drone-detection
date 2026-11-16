#!/usr/bin/env python3
"""Build a Colab continuation using the prior complete BEATs study cache."""

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
        raise ValueError("Pin the complete published implementation commit")
    previous = json.loads((ROOT / "notebooks/public_corpus_comparison.ipynb").read_text())

    def prior(i):
        return "".join(previous["cells"][i]["source"])

    cells = [
        cell(
            "markdown",
            """
# BEATs mechanical-confuser comparison

This continues the **complete** previous public-corpus study, using its existing
BEATs embeddings and trained head. It compares the original detector, a broader
decision threshold, and a head trained with additional mechanical negatives.
The pretrained BEATs encoder remains frozen.

The previous test is consumed development data. The new public challenge uses
all eligible FSD50K evaluation recordings, supported WSU flight/background
intervals, and indexed AuDroK point recordings. Publisher labels are weak;
three-second crops are not independent field trials or false-alarm/hour exposure.

Use the same CPU/GPU setting as the previous full study. Its complete BEATs
checkpoint, frozen head, and all222 original archive receipts must exist. This
notebook verifies those inputs before downloading the additional6.26GB FSD
audio. It processes the complete reserved cohort and does not substitute a
small demonstration when inputs are missing. New results and model parameters
persist in a separate Drive project. A published evaluation cohort is consumed;
rerunning it is a reproduction, not another fresh generalization test.
""",
        ),
        cell(
            "code",
            f"""
ARCHIVES_DRIVE_PATH = '/content/drive/MyDrive/drone-detection-v2/corpus-intake-v3/source-archives'
PREVIOUS_PROJECT_DRIVE_PATH = '/content/drive/MyDrive/drone-corpus-study-v1-20261115'
PROJECT_DRIVE_PATH = '/content/drive/MyDrive/drone-beats-confusers-v1-20261116'
REPO_URL = 'https://github.com/StoneDerek/drone-detection.git'
CODE_COMMIT = '{args.commit}'
DEVICE = 'cuda'  # Must match the previous complete study's BEATs encoder identity.
FEATURE_SHARDS = 1
""",
        ),
        cell("code", prior(2)),
        cell("code", prior(3)),
        cell("code", prior(4)),
    ]
    preparation = prior(5).replace(
        "'/content/drone-corpus-study-'", "'/content/drone-confuser-previous-'"
    )
    preparation = preparation.replace(
        "MODEL_RESULTS = PROJECT / DEVICE / 'models'",
        "MODEL_RESULTS = Path(PREVIOUS_PROJECT_DRIVE_PATH).resolve() / DEVICE / 'models'",
    )
    # Fail before costly preparation/downloads if the complete prior study is absent.
    cells.append(
        cell(
            "code",
            """
PREVIOUS_PROJECT = Path(PREVIOUS_PROJECT_DRIVE_PATH).resolve()
if not PREVIOUS_PROJECT.is_relative_to(MOUNT) or PREVIOUS_PROJECT == PROJECT:
    raise ValueError('Choose separate mounted Drive projects for previous and new results')
for path in [PREVIOUS_PROJECT / DEVICE / 'models/beats/model.json',
             PREVIOUS_PROJECT / DEVICE / 'models/beats/head.npz',
             PREVIOUS_PROJECT / DEVICE / 'feature-checkpoints/beats.tar.gz']:
    if not path.exists():
        raise RuntimeError('The previous full BEATs study is required: ' + str(path))
"""
            + preparation,
        )
    )
    cells.append(
        cell(
            "code",
            """
PREVIOUS = STUDY
run_script('checkpoint_corpus_features.py', 'restore', '--directory', PREVIOUS / 'features/beats',
           '--checkpoint', PREVIOUS_PROJECT / DEVICE / 'feature-checkpoints/beats.tar.gz')
old_model = json.loads((PREVIOUS / 'models/beats/model.json').read_text())
cached_encoder = json.loads((PREVIOUS / 'features/beats/encoder.json').read_text())
if old_model['encoder'] != cached_encoder or cached_encoder['device'] != DEVICE:
    raise ValueError('Previous model, cached embeddings and chosen execution device must agree')
ASSETS = PROJECT / 'model-assets/beats'
run_script('fetch_beats.py', '--output', ASSETS)
WORK = PROJECT / DEVICE / 'confuser-study-v1'
WORK.mkdir(parents=True, exist_ok=True)
run_script('acquire_confuser_study.py', '--work', WORK, '--metadata-only')
run_script('prepare_confuser_study.py', 'plan', '--work', WORK, '--previous', PREVIOUS, '--urban-metadata', URBAN)
if not (WORK / 'models/frozen-models.json').exists():
    run_script('run_confuser_study.py', 'fit', '--work', WORK, '--previous', PREVIOUS)
print('All three variants frozen before fresh audio scoring:', WORK / 'models/frozen-models.json')
""",
        )
    )
    cells.append(
        cell(
            "code",
            """
run_script('acquire_confuser_study.py', '--work', WORK)
for stage in ['wsu', 'audrok', 'fsd', 'seal']:
    run_script('prepare_confuser_study.py', stage, '--work', WORK)
run_script('run_confuser_study.py', 'authorize', '--work', WORK)
FRESH_FEATURES = WORK / 'features/beats'
CHECKPOINT = PROJECT / DEVICE / 'feature-checkpoints/beats.tar.gz'
run_script('checkpoint_corpus_features.py', 'restore', '--directory', FRESH_FEATURES, '--checkpoint', CHECKPOINT)
run_script('extract_corpus_study_features.py', '--prepared', WORK / 'prepared', '--output', FRESH_FEATURES,
           '--family', 'beats', '--assets', ASSETS, '--device', DEVICE, '--threads', '1', '--splits', 'final')
run_script('checkpoint_corpus_features.py', 'pack', '--directory', FRESH_FEATURES, '--checkpoint', CHECKPOINT)
""",
        )
    )
    cells.append(
        cell(
            "code",
            """
REPORT = PROJECT / DEVICE / 'report'
run_script('run_confuser_study.py', 'evaluate', '--work', WORK, '--report', REPORT)
import pandas as pd
display(pd.read_csv(REPORT / 'metrics.csv'))
print('Comparison:', REPORT / 'comparison.json')
print('Per-crop scores:', WORK / 'models/final-predictions.jsonl')
print('Complete source/quality receipts:', WORK / 'prepared/preparation-complete.json')
print('The cohort is now consumed. Do not tune on it and call the next run fresh evaluation.')
""",
        )
    )
    result = dict(
        cells=cells,
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python"),
            colab=dict(provenance=[], name="BEATs mechanical-confuser comparison"),
        ),
        nbformat=4,
        nbformat_minor=5,
    )
    for i, c in enumerate(result["cells"]):
        c["id"] = f"confuser-{i}"
        if c["cell_type"] == "code":
            compile(c["source"], f"cell{i}", "exec")
    (ROOT / "notebooks/beats_confuser_comparison.ipynb").write_text(
        json.dumps(result, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
