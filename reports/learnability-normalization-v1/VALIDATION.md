# Validation and provenance

- **88 Python tests pass**, including the existing suite and five new numerical
  guards. Ruff and Git whitespace checks pass. The pre-existing Keras input-name
  warning remains informational.
- Exact RMS is a new explicit diagnostic mode. Existing boolean normalization
  keeps its prior formula and default serialized configuration. Tests verify
  exact RMS, intentionally unclipped float peaks, silence, unknown modes,
  unchanged bounded gain, permutation marginals and train-only interval/class
  weighting. Synthetic numerical probes are not acoustic trials.
- All **5,498 publisher file hashes** in the ITU release and all four cached
  tensor array SHA-256 values were freshly checked. The source session, group
  and split identities remain pinned. No new listening judgments or audible
  absence labels were assigned.
- Eight CNN runs have the same **154,689 parameters**. Initial weight hashes
  match across all amplitude arms for each seed. Batch orders, sample weights,
  optimizer, epoch budget and validation selection are fixed. Exact-RMS is a
  disclosed input-coverage follow-up; it is not an accuracy-selected architecture.
- Complete selected and last numeric arrays, training histories, source
  snapshots, input identities and paired results are retained. **72 prespecified
  real-input scores** reproduce from the standalone packages within the declared
  tolerances. The non-neural packages include their full feature configuration,
  train-only means/scales and numeric coefficients.
- Both existing CNNs’ scores on all eight contexts match their saved pre-change values
  within1e-5 during the band-occlusion follow-up. The SVM's time-permutation
  invariance is independently checked on actual inputs. These are software and
  sensitivity checks, not another external assessment.

The original six-arm learning process imported the original source before the
exact-mode addition. `original-execution-source.tar.gz` and `execution.py` preserve
those exact bytes. The exact-RMS follow-up has its own execution/source archive
and source hashes. The maintained training entrypoint subsequently removes an
unused import; archived executed source remains byte-identical to its record.

Per-arm elapsed times include concurrent diagnostic/learning processes. They
are compute records, not controlled performance or live-latency benchmarks.
Every source is already consumed for development. No result here is independent
field validation, and no diagnostic model is promoted to the product.

## Inspect and reproduce

`scripts/summarize_learnability_v1.py` recreates the tables and learning curves
from retained reports. `scripts/verify_learnability_v1.py` checks numeric package
inference when the pinned source clips and ignored score arrays are available.
The four source/cached-input identities are in `input-integrity.json`.

For a fresh reproduction of training, use an isolated directory and the archived
execution source with the pinned ITU release. The completed worktree deliberately
preserves finished arms and rejects changed execution/cache identities. Repeating
these consumed clips does not create new independent evidence.
