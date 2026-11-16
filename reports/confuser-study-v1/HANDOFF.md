# Completed BEATs confuser milestone

Branch: `codex/beats-confusers-v1-20261116` in
[StoneDerek/drone-detection](https://github.com/StoneDerek/drone-detection/tree/codex/beats-confusers-v1-20261116).
The main branch was not changed.

The numerical implementation used for the fixed experiment is pinned at
[`d47f094a2b5eb967cc2dcd9d66c28bd3eabaac3d`](https://github.com/StoneDerek/drone-detection/commit/d47f094a2b5eb967cc2dcd9d66c28bd3eabaac3d).
The [Colab continuation](https://colab.research.google.com/github/StoneDerek/drone-detection/blob/9f510e78b856ff4d8adc4dd4783df1a00650cd9b/notebooks/beats_confuser_comparison.ipynb)
pins that exact commit. It requires the previous complete BEATs study cache,
uses the same explicit CPU/GPU identity, and processes the whole reserved
cohort. The notebook was structurally validated and its code cells compiled;
the full new experiment was executed here on CPU, not in Colab.

The encoder is pretrained. Both the previous and newly trained task heads have
learned drone/non-drone labels; neither is an untrained random detector. This
milestone did not fine-tune the encoder weights. Cached previous embeddings
make linear-head fitting quick. New audio decoding, checksum verification and
9,665 accepted-crop embeddings required the substantial computation.

Read [FINDINGS.md](FINDINGS.md) for interpretation and the next experiment,
[RESULTS.md](RESULTS.md) for every primary rate, and
[PROTOCOL.md](PROTOCOL.md) for the prespecified choices. The unusually poor
WSU background result is included in both the table and the final chart.
[data-quality.json](data-quality.json) accounts for all 24 exclusions;
[source-volume-verification.json](source-volume-verification.json) preserves
the distinction between whole-volume and bounded verification.

The local result bundle includes complete predictions, manifests, the reserved
plan, source receipts, model parameters, licenses/credits and a separate
researcher-only field-review key. No raw audio or pretrained encoder weights
are included. Its SHA256 and size are in [bundle-receipt.json](bundle-receipt.json).
Public numeric heads are in [artifacts/confuser-study-v1](../../artifacts/confuser-study-v1).

The [blind field-review inventory](blind-field-review.csv) covers all 1,734
selected field crops and contains anonymous review IDs and empty annotation
columns. It omits internal paths, source filenames, model predictions and publisher classes.
The local bundle contains the separate full playback inventory. Keep the researcher
key and error tables away from a blinded reviewer. Paths refer to this prepared
corpus; another runtime must reconstruct the audio using the pinned pipeline.

Software validation: 185 tests passed, eight optional tests skipped; source,
scripts and tests passed Ruff. Both software-checks and notebook-recovery CI
jobs passed on the pre-result publication. Report rendering was additionally
checked against the actual completed comparison. These software checks do not
validate the detector's acoustic accuracy.
