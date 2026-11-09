# Experimental available-context gain model

**Development only.** This candidate has no fresh independent assessment. It is
optional and does not replace the default in `../real-v2/candidate`.

`candidate/` is a complete frozen package: model metadata, classifier weights,
train-only means/scales and official YAMNet encoder with hashes. Product 0.2.0
implements its `yamnet_rms` feature path. After installing that product version,
inspect your own recording with:

```bash
drone-detector serve --bundle candidate
```

The interface prominently identifies this model as a development experiment.
Its selected on/off thresholds are 0.8/0.6 with one on-hit and two off-hits. Do
not copy these thresholds onto the default model or add gain normalization in
front of the default model's weights.

[Development results](../../reports/context-gain-v1/RESULTS.md) compare 21/22
positive files detected and 9/188 negative files alarmed against 20/22 and 9/188
for the default. Those are correlated publisher-labeled development recordings,
not a new field-reliability benchmark.

Reproduce from the repository root after preparing the pinned real-data corpus:

```bash
.venv/bin/python scripts/experiment_context_gain.py
```

The script preserves an existing experiment, verifies the manifest identity and
does not open a fresh assessment. The exact training source fingerprint is
preserved in `training-source.tar.gz` and `training-source.json`. Development
selection used the later UI-only source revision in
`../product-v0.2/product-source.tar.gz`; both identities are recorded in the
model metadata and control verification. Feature/training logic is unchanged
between those snapshots.

`SHA256SUMS` covers this experimental package. The original code is MIT; the
included YAMNet model retains its Apache-2.0 attribution. Raw third-party audio
is not redistributed. No acoustic range, wind tolerance, SNR or event-reliability
claim is made.
