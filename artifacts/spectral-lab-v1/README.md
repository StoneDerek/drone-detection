# Spectrogram model lab 0.3.0

This package supplies six initial CNNs, frequency-context and matched-axis CNN follow-ups,
a fully non-neural RBF SVM,
their original coarse-grid versions, numeric weights/statistics, source
snapshots and the local comparison/Grad-CAM interface. All new models are
**development experiments**, with no fresh independent acoustic assessment.
The previously assessed YAMNet default remains in `../real-v2/candidate`.

From the repository, run:

```bash
bash scripts/setup.sh
bash scripts/start_lab.sh
```

Choose audio, compare models, play/seek the timeline, and select a CNN context,
class logit and convolution layer for Grad-CAM. Frequency zoom changes only the
display. Training histories and selected epochs are visible. Comparison and
explanation JSON include model identities and actual tensor dimensions. The
display matrices are quantized to eight bits; model inputs use float32.
Upload comparison is limited to two minutes/64 MB. It does not require the
training/evaluation corpus, cached features or network weight downloads.

For an isolated Python 3.12 install, from this directory:

```bash
python3.12 -m venv /tmp/drone-lab
uv pip sync --python /tmp/drone-lab/bin/python --require-hashes runtime-requirements.txt
uv pip install --python /tmp/drone-lab/bin/python --no-deps acoustic_drone_detector-0.3.0-py3-none-any.whl
/tmp/drone-lab/bin/drone-detector lab --catalog catalog.json
```

Keep `../real-v2/candidate` and `../real-v2/baseline` alongside this folder for
the existing controls in the catalog. The new CNN and SVM packages themselves
are complete within `models/`. `coarse-grid/` preserves the initial decisions;
`models/` contains explicitly additional development calibration, using the
same neural/SVM parameters. No failed arm was removed or relabeled as a success.

Reproduce experiments after preparing the pinned public corpus:

```bash
.venv/bin/python scripts/experiment_spectrograms.py
.venv/bin/python scripts/experiment_frequency_context.py
.venv/bin/python scripts/experiment_matched_frequency_axis.py
.venv/bin/python scripts/calibrate_spectral_models.py
.venv/bin/python scripts/package_spectral_lab.py
```

The original plan, implementation optimization/parity check, complete results,
per-source errors, experiment ledger and compressed development score traces
are retained in [the report](../../reports/spectral-lab-v1/RESULTS.md).
The [architecture/data account](../../docs/model-and-data.md) distinguishes the
new baseline from Shikang's historical EfficientNet model and its much larger
reported training corpus. His checkpoint has not been benchmarked here.

The original CNN has 1,781 parameters and accepts frequency × time × one-channel tensors
without resizing them into a fixed image. All six use the same three convolution
blocks, global average pooling, dropout and scalar head. The separate 3,077-parameter
follow-up adds one frequency-only convolution to span more adjacent harmonics.
Genuine Grad-CAM
requires the selected trained convolution layers. It is not available for the
SVM or the opaque YAMNet SavedModel; none is fabricated. A heatmap does not prove
that a model learned a useful or causal cue. The lab discloses native attribution
resolution and zero maps.

`product-source.tar.gz`, the wheel, locked runtime requirements and source/model
fingerprints support reproduction. `SHA256SUMS` covers this package; the original
default is covered by its unchanged `../real-v2/SHA256SUMS`. Code is MIT; the
YAMNet control retains its third-party attribution. Raw public audio and large
feature caches are not redistributed.
