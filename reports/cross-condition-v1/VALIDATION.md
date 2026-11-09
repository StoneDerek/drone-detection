# Validation and reproduction

The complete Python suite passes: **96 tests**, with the existing informational
Keras input-name warning. Ruff passes. Eight new guards cover complete-context
sampling, primary-versus-secondary publisher labels, fixed package mutations,
positive-only metrics, original-member hashes, lossless PCM, rate-limit retries
and pooling undocumented capture dates before model use. Numerical fixtures are
software tests, not additional acoustic evidence.

**11,180 recording outcomes reproduce**: 11,160 classifier decisions and 20
original-policy legacy alarms. **336 direct real-input scores** pass their
package tolerances, with maximum error below 1e-5. All 273 retained PCM files and
their original publisher JSONs pass the final identity checks. All four reused
earlier tensor-cache hashes match their original records.

`verification.json` records the full trace-decision reapplication, direct numeric
package checks on fixed real inputs, and retained PCM/metadata checks. Its
software repetitions do not add independent acoustic trials. The verifier
reconstructs each recording's decision, maximum, positive-score fraction and
source table from every retained score trace, including the original-policy
legacy alarm decisions. Direct checks choose the first recording in each
source/class cell and its first/middle sampled context. CNN scores must agree
within 1e-5; descriptor-logistic scores within 1e-8.

The one-channel source's original selected WAV and JSON SHA-256 values match
the publisher manifest; the file index must agree. ZIP reads check member CRCs.
Every retained FLAC is decoded to signed 16-bit PCM and compared bit-for-bit
with the original WAV before its receipt is accepted. The final verifier checks
the stored PCM hashes, original JSON bytes, planned original hashes and amended
group assignments again. Partial fetching does **not** verify the whole archive
MD5. Kenya's five WAVs and three metadata files match their pinned Git blobs.
Whole-recording fingerprints found no cross-corpus matches; shifted/lossy
derivatives are not comprehensively ruled out.

The model panel and each source-use ledger retain their weights, preprocessing,
thresholds, source revisions, code identities and selected recording hashes.
The first Nanjing attempt hit HTTP429. Completed receipts were preserved; larger
ranges and fewer workers resumed with server backoff. `fetch-history.json` and
three source archives preserve the initial, serial-resume and final fetch code.
This transport change selects no different audio and changes no model. The
metadata-only grouping amendment predates that source's first-use ledger and
changes no audio selection, label, weight, threshold or sampling rule.

The descriptor contrast is the exact standardized linear-model contribution
to positive-minus-negative mean logit, with equal publisher-group/file mass
within class. It is not the maximum-score file decision. The 18-context paired
probe is a disclosed follow-up on consumed inputs, with all twelve models and
all transformations retained. Its independent time permutations are nonphysical
inputs. Raw logits expose changes hidden by saturated probabilities. The bounded
gain arithmetic audit confirms all quieter inputs remain above the silence guard;
only two of eighteen original/quiet pairs both reach the nominal RMS target.
No listening judgments, temporal absence labels or acoustic causal claims follow.

The retained reports can be summarized without rerunning audio inference:

```bash
.venv/bin/python scripts/summarize_cross_condition_v1.py
```

When the pinned ignored raw inputs are present, software verification is:

```bash
.venv/bin/python scripts/verify_cross_condition_v1.py
```

Completed first-use cohorts deliberately reject an inference rerun that would
overwrite the record. Reproduce their inference in an isolated checkout using
the archived source, fixed plans and pinned publisher inputs. For the controlled
source, restore publisher indexes from `controlled-source/source-records/` to the
documented ignored metadata directory, fetch the fixed members, apply the
committed grouping amendment, and evaluate the unchanged panel. The raw source
audio, cached tensors and fetch logs stay under ignored `runs/`; complete model
packages, raw score traces, label/index snapshots and results are committed.

All inspected audio is now consumed for future tuning. No model package,
architecture, public deployment or product default was changed.
