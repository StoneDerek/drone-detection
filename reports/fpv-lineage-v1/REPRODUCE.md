# Reproducing the complete FPV audit

Use the repository's locked Python environment (`bash scripts/setup.sh`) and
FFmpeg/ffprobe. The observed audit used FFmpeg 7.1.5, libsndfile via SoundFile 0.13.1
and NumPy 2.3.5. AAC decoder changes can change exact sample identity, so retain the
decoder version and do not turn a decoder mismatch into a confident no-overlap claim.
The scripts run no publisher code and no detector.

Use an isolated reproduction directory. The intake file pins both the publisher
MD5 and the observed SHA-256. Its profile is only an acquisition selection, not a
training profile. Downloading needs about 2.16 GB, in addition to runtime and ledger
space; the audio is decoded in memory rather than extracted permanently.

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python scripts/fetch_corpus_intake.py \
  --manifest reports/fpv-lineage-v1/intake.json --profile fpv-lineage \
  --output work/fpv-reproduction-v1/archives --budget-gib 2.1

.venv/bin/python scripts/prepare_fpv_lineage_inputs.py \
  --archive work/fpv-reproduction-v1/archives/zenodo-21935566/Fpv.zip \
  --output work/fpv-reproduction-v1/metadata

.venv/bin/python scripts/audit_fpv_lineage.py \
  --archive work/fpv-reproduction-v1/archives/zenodo-21935566/Fpv.zip \
  --metadata work/fpv-reproduction-v1/metadata --work work/fpv-reproduction-v1

.venv/bin/python scripts/verify_fpv_direct_lineage.py \
  --archive work/fpv-reproduction-v1/archives/zenodo-21935566/Fpv.zip \
  --work work/fpv-reproduction-v1

.venv/bin/python scripts/audit_fpv_split_overlap.py --work work/fpv-reproduction-v1

.venv/bin/python scripts/link_fpv_annotations.py \
  --metadata work/fpv-reproduction-v1/metadata --work work/fpv-reproduction-v1

.venv/bin/python scripts/audit_fpv_raw_links.py \
  --archive work/fpv-reproduction-v1/archives/zenodo-21935566/Fpv.zip \
  --metadata work/fpv-reproduction-v1/metadata --work work/fpv-reproduction-v1
```

For the exact prior-identity screen, the current study manifest must exist locally
and match the manifest SHA-256 recorded in `audit.json`. Historical snapshots are
already in the repository. The current study's original-archive preparation is
documented in [the BEATs handoff](../beats-adaptation-v1/HANDOFF.md). This stage reads
recorded byte identities; it does not require opening the historical audio paths.

```bash
.venv/bin/python scripts/audit_fpv_prior_identity.py \
  --work work/fpv-reproduction-v1 \
  --manifest work/beats-adaptation-cpu2-v1/manifest.jsonl \
  --manifest reports/real-v2/manifest-snapshot.jsonl \
  --manifest reports/external-v3/recordings.jsonl \
  --manifest reports/cross-condition-v1/controlled-source/recordings.jsonl \
  --manifest reports/cross-condition-v1/transfer-recordings.jsonl \
  --manifest reports/cross-condition-v1/kenya-recordings.jsonl
```

The original acquisition receipt was written by the acquisition call and copied
to the audit work directory. Preserve the corresponding receipt from a fresh run
for aggregate publication:

```bash
cp work/fpv-reproduction-v1/archives/zenodo-21935566/Fpv.zip.complete.json \
  work/fpv-reproduction-v1/acquisition-complete.json
.venv/bin/python scripts/summarize_fpv_lineage.py \
  --archive work/fpv-reproduction-v1/archives/zenodo-21935566/Fpv.zip \
  --metadata work/fpv-reproduction-v1/metadata --work work/fpv-reproduction-v1 \
  --output work/fpv-reproduction-v1/published-audit.json
```

The main audit verifies the full archive and equality of its central directory
with the recovered catalogue. Each decoded member is fully read, checking ZIP CRC.
The aggregator rejects duplicate rows and incomplete accounting. The recovered
metadata JSON formatting and intake receipt differ from the earlier range-audit
snapshot, so their file hashes can differ while parsed source records are equal.
Do not compare hashes from different serialization/receipt stages as if they were
waveform identity hashes. Per-member native SHA-256s and fixed-contract sample
matches are the audio evidence.

Full direct matching uses native float32 bytes at the same sample rate and channel
count. Unique offsets are required; repeated identical segments remain ambiguous.
The overlap stage only assesses verified edges, and the annotation stage preserves
unmatched or ambiguous cases. No timing, label or independent-session assumption
fills those gaps.
