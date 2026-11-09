# Product update 0.2.0

This wheel adds signal-quality advisories to local upload analysis and JSON
score traces. The timeline distinguishes **Review audio quality**, **Silent
input**, **Insufficient audio**, **No alarm**, and **ALARM**. An active alarm
takes priority over an advisory. Playback, seeking and JSON export remain
available.

The default model, its weights, train-only statistics and alarm thresholds are
unchanged. Use the complete frozen bundle in `../real-v2/candidate`; its existing
[assessment and limits](../../reports/real-v2/RESULTS.md) still apply. The optional
context-gain model in `../context-gain-v1/candidate` is a development experiment
with no fresh independent assessment.

From this directory, install into a separate Python 3.12 environment:

```bash
python3.12 -m venv /tmp/drone-product-02
uv pip sync --python /tmp/drone-product-02/bin/python --require-hashes runtime-requirements.txt
uv pip install --python /tmp/drone-product-02/bin/python --no-deps acoustic_drone_detector-0.2.0-py3-none-any.whl
/tmp/drone-product-02/bin/drone-detector serve --bundle ../real-v2/candidate
```

Keep this folder and `../real-v2/candidate` together when copying the package.
Inference requires no evaluation corpus, notebook, editable checkout or weight
download after installation. The repository's `bash scripts/start.sh` uses this
same default model.

`product-source.tar.gz` preserves the updated source, lockfile, scripts and
software checks. `source.json` records the source fingerprint. `SHA256SUMS`
covers this update's files; the original model files remain covered by the
unchanged `../real-v2/SHA256SUMS`.

See [validation](../../reports/product-v0.2/VALIDATION.md) and the
[gain experiment](../../reports/context-gain-v1/RESULTS.md). Quality advisories
are signal measurements, not validated wind, SNR, acoustic range or field
reliability estimates. Generated tones are used only to check software states;
real recordings remain the acoustic evidence.
