# Validation and scope

- The committed starting version passed all 80 existing Python tests. The final
  branch passes **83 tests** and Ruff. Three additional checks independently
  verify one-second FFT descriptors without padding, reject false waveform
  continuity, and reject changes to a frozen panel's alarm rule. Synthetic
  numerical probes are software evidence only.
- The source download verifies all 37 FPV Git blobs and README, the ITU archive
  publisher MD5, and 5,498 listed ITU SHA-256 entries. Raw publisher bytes remain
  outside tracked source. The 2,356-record old corpus supplied all three duplicate
  fingerprint types for overlap checks.
- Recovered CNN arrays match their deployed SHA-256 values. Architecture,
  original model identity and decisions match committed reports. All 32
  pre-existing numeric fixture comparisons pass within 1e-5. Rewrapped metadata
  receives new identities, recorded separately from the original model IDs.
- The first complete external recording for each spectral model is scored by
  both batched and sequential replay. Score errors are at most 1e-5, and alarm
  states match. The exact parity measurements are in the per-model JSON files.
- All **4,288 model/recording outcomes** reproduce exactly by reapplying their
  frozen decisions to the retained timestamped score traces. This is replay
  verification, not a second acoustic assessment. The one-use ledger is written
  before external scoring and refuses another assessment in place.
- The small task retains all 55 publisher groups in one split each and finds no
  exact half-second raw sample fingerprint crossing splits. Train-only numeric
  parameters reproduce six prespecified real-input scores within 1e-10.
- Recovered models and the small classifier are preserved as non-executable
  numeric NPZ/JSON packages, with SHA256SUMS. The exact executed source is
  archived alongside them. No training on the DDL precision repair occurred.

The original YAMNet product, public deployment and environment publication are
not changed. The two external sources are now consumed for any subsequent model
optimization. Weak/published labels, single-session dependence, missing continuous
negatives and incomplete audibility truth prevent a field-reliability claim.

## Reproduction

The branch contains the frozen panel, compatibility amendment, audited manifest,
source metadata, compressed scores, full outcomes, prespecified review cases and
the separate small-task report. Reproduce the saved decision outcomes with:

```bash
bash scripts/setup.sh
bash scripts/check.sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/summarize_external_v3.py
```

Raw-source recovery uses pinned originals, not changing URLs or substitute audio:

```bash
mkdir -p runs/external-v3/research
cp reports/external-v3/source-records/* runs/external-v3/research/
.venv/bin/python scripts/fetch_external_v3.py
```

The source record filenames required by the fetcher are `fpv-tree.json` and
`itu-zenodo.json`. Recreate the two ignored execution-path model directories from
`artifacts/external-v3/recovered/` when needed. Repeating a scored input does not
create fresh independent evidence. A clean clone already contains this completed
assessment's ledger; the original one-use runner intentionally refuses to
overwrite it. The retained execution snapshot records exactly how it was run.
