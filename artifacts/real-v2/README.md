# Real-audio product package

This is an experimental recording-presence detector, not a field-certified alarm.
The [fresh assessment](../../reports/real-v2/RESULTS.md) includes failed cases and
states which provisional recording targets passed. Event recall, true false
alarms per hour, acoustic range and live capture remain unvalidated.

`candidate/` contains the development-selected model; `baseline/` contains the
version-2 relative-Mel comparison. Each package has `model.json` and
`parameters.npz`: class mapping, context/hop/sample rate, frozen decisions,
train-only feature means/scales, coefficients/intercept, data/code identities,
all development threshold trials and provenance. Audio-pretrained packages also
contain the complete frozen YAMNet SavedModel and file hashes. No training
notebook, omitted data module or substitute normalization file is needed.

Loading validates SHA-256, parameter dimensions, finiteness and class mapping.
NPZ uses `allow_pickle=False`. `SHA256SUMS` covers the distributed package files;
these are integrity checks, not third-party authenticity signatures.

The wheel and hashed runtime requirements support an isolated install:

```bash
python3.12 -m venv /tmp/drone-product
/tmp/drone-product/bin/python -m pip install --require-hashes -r runtime-requirements.txt
/tmp/drone-product/bin/python -m pip install --no-deps acoustic_drone_detector-0.1.0-py3-none-any.whl
/tmp/drone-product/bin/drone-detector serve --bundle candidate
```

No evaluation corpus or network connection is required for file inference after
installation. The optional benchmark examples require the separately fetched
audio and its matching manifest. See the repository README for preparation and
the product startup script. Inference validation evidence is saved with the
assessment; it does not constitute new independent acoustic evidence.

`evaluated-source.tar.gz` preserves the training/inference source snapshot at the
freeze. The base Git HEAD alone predates these changes; the recorded source
content fingerprint identifies the actual implementation. Data versions,
parent groups, labels and hashes are retained in the report's manifest snapshot.
Rerunning training can change model IDs because metadata contains measured wall
time; frozen weights and scores remain directly reviewable.

Code is MIT. The included Google YAMNet release, when present, retains its
third-party attribution/license. Data sources and their separate terms are
documented in the report. Raw third-party recordings are not bundled here. Fresh
ESC-50 categories under CC BY-NC were used only for assessment, not to fit model
parameters. Published recording labels and origin metadata were not independently
verified for every interval, and clip counts do not establish independent trials.
