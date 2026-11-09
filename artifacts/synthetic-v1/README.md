# Experimental synthetic model package

**Not a validated drone detector.** These models were trained only on generated
harmonic/noise signals. No authentic drone capture, microphone, range or field
condition is supported. The candidate misses one of ten held-out generator
events. See [all results](../../reports/synthetic-v1/RESULTS.md).

- `baseline/`: absolute mean/std log-Mel logistic classifier, on threshold 0.65,
  off 0.4875, one high score to activate, two low scores to release.
- `candidate/`: relative log-Mel, train-only scalar-gain augmentation seed 42,
  on threshold 0.8, off 0.6, one high score to activate, two low scores to release.
- Each directory contains `model.json` and `parameters.npz`. The latter includes
  **all** training-only means, scales, coefficients and intercept. JSON records
  class mapping (0 = non_drone, 1 = drone), feature configuration, training
  recording IDs/hashes/window counts, versions, code fingerprint, frozen contract
  and every development threshold trial. No notebook or missing data/ module is
  required. No statistics may be replaced with stand-ins.
- `evaluated-source.tar.gz` preserves the exact Python sources, then-current
  project/lock and contract used for the reported frozen synthetic comparison.
  `evaluated-source.json` records its source fingerprint and base Git HEAD. Its
  source fingerprint was verified against both model JSONs. The Git HEAD alone
  is the original empty repository; the content fingerprint identifies the
  implementation before it is committed.
- `acoustic_drone_detector-0.1.0-py3-none-any.whl` is the portable inference/tooling
  package tested in a clean virtual environment. It also includes post-evaluation
  validation of minimum-hop configuration, demo manifest/content checks, the
  failure appendix and a development-claim guard. These changes did not change
  the frozen model, features or alarm decisions. Its scores were checked against
  the frozen reference trace; do not treat this compatibility check as a fresh
  holdout assessment.
- `runtime-requirements.txt` exports the exact pinned runtime dependencies and
  distribution hashes. `SHA256SUMS` covers the package, arrays, metadata,
  requirements and evaluated source snapshot. Hashes detect changes; they are
  not signatures asserting trust in a third-party model.

From this directory, verify `sha256sum -c SHA256SUMS`. On a supported Python 3.12
host, install the hashed dependencies into a virtual environment, then the wheel:

```bash
python3.12 -m venv /tmp/drone-inference
/tmp/drone-inference/bin/python -m pip install --require-hashes -r runtime-requirements.txt
/tmp/drone-inference/bin/python -m pip install --no-deps acoustic_drone_detector-0.1.0-py3-none-any.whl
/tmp/drone-inference/bin/drone-detector infer \
  --bundle candidate --audio /path/to/recording.wav --output /tmp/scores.jsonl
```

No network access is needed for inference once dependencies and this package
are installed. Inference must not be interpreted as a real drone presence claim.
The clean-venv test used a copied package and WAV from an unrelated working
directory with isolated Python (`-I`) and no editable source import. It is a
same-host clean installation test, not a newly provisioned machine test.

Manifest snapshots and split definitions live in `reports/synthetic-v1`.
`manifest-snapshot.jsonl` is an immutable **metadata snapshot**, not an audio
directory: its relative paths refer to the regenerated corpus root. Regenerate
the synthetic corpus using the README command; its manifest SHA must match
`9a3a26196c2ad0d63893cab939803e00fd612e3c29978815eabebce45df71e68`.
Only baseline and candidate packages are retained here; the experiment command
regenerates every ablation package. Timing metadata makes regenerated model IDs
potentially different even when predictions match.
