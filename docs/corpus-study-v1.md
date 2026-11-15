# Public-corpus representation study

This milestone replaces the earlier five-recording software fixture with an
actual corpus comparison. It uses all eligible files from the eight pinned
releases in `data/corpus-intake-v3.json`, with explicit exclusions and shared
audio inputs for every classifier. Results belong in
`reports/corpus-study-v1/RESULTS.md` after execution completes; neither a successful
download nor passing software tests is a model result.

## Question and scope

Do broadly pretrained audio encoders support more transferable drone
classification than the project's simple spectral summaries when fresh heads
receive the same real training material?

The first comparison is deliberately bounded: one fixed central three-second
crop per eligible native file, or the actual shorter signal if it is at least
250 ms long. It does not exhaust every second of long recordings. Native-file
hours, selected crop-hours, original recording IDs and campaign groups are
reported separately. This is weak-label crop classification, not a claim to
meet the proposed continuous-detector targets.

| Arm | Representation | Fitted parameters |
|---|---|---|
| Spectral | Existing relative-harmonic summaries | Standardizer and linear logistic head |
| CLAP | Pinned frozen LAION audio encoder, 512 dimensions | Standardizer and linear logistic head |
| BEATs | Pinned frozen pretrained BEATs encoder, 768 dimensions | Standardizer and linear logistic head |
| Nuisance control | Crop duration, log RMS, log peak, near-zero fraction | Standardizer and linear logistic head |

All heads use fixed C=1, adequate convergence checks, TRAIN-only weighted scaling,
and weights that balance class, source, capture group, and rows within groups.
This comparison does not fine-tune an encoder, search architectures, or attribute
every difference solely to pretraining. Each encoder retains its released native
frontend. CLAP's 16-to-48 kHz resampling restores no information above 8 kHz.
BEATs uses a checksum-pinned mirror whose equivalence to the unavailable official
OneDrive bytes has not been independently verified.

## Prospective partition

| Partition | Positive material | Negative-label material |
|---|---|---|
| TRAIN | Poland: 15 March, 15 April, 16 April 2023; Berne: 3 August 2022 | TUT2017 development; DNC |
| DEVELOPMENT | Poland: 17 April 2023 | TUT2017 evaluation |
| FINAL | Poland: 18 April 2023 at Lapalice; Edzell DroneNoise; processed NEAPTIDE | UrbanSound8K |

No source in this table was chosen by this study's detector scores. Capture
dates and derivative parent IDs remain together. DNC's unresolved provenance
collapses to one conservative group. TUT parent IDs are explicitly checked for
overlap between releases. UrbanSound8K excerpts retain their Freesound parent
and offsets; its excerpt count is not its independent recording count.
Parents shared with historically consumed ESC-50 material are quarantined
before final use, even though these fresh heads do not train on ESC-50.

All Polish speech-plus-drone measurements remain positive. These measurements
add vehicle and recording variety but primarily contain hovering at a fixed
measurement grid. Berne and Edzell provide different flight conditions.
NEAPTIDE includes acoustic corrections and is identified separately as a stress
set. Final positives cover few independent campaigns; no confidence interval
will treat their correlated files as independent field trials.

The fixed physical-channel policy uses channel 0 in native multichannel files,
M1 in DroneNoise and microphone 1 in NEAPTIDE. Calibration files and other
simultaneous views are excluded. Two Edzell event prefixes in this intake have
only M5, so they do not enter the M1 comparison. Their exclusion is unrelated to
model behavior. The exact release manifest, rather than older approximate
publisher counts, determines the files acquired.

## Data checks and final-use controls

Acquisition verifies the publisher's distribution checksum and size, computes
stored SHA-256, and preserves resumable partial state. Preparation rechecks
archive SHA-256 and fully reads each selected ZIP member for CRC verification.
Canonical float32 audio preserves gain, selects a physical channel, and uses
polyphase resampling to 16 kHz. It does not manufacture additional examples from
channels, repeat crops or augmentations.

Before fitting, quality finalization excludes crops intersecting TUT's published
fault intervals, checks exact native/canonical duplicates, and verifies candidate
gain copies by waveform residual. Copies spanning partitions or labels are all
quarantined. Same-partition copies retain one deterministic representative.
Shifted or lossy cross-source excerpts and pretrained-model overlap are not
fully ruled out. Feature caches may contain subsequently excluded crops; model
fitting and evaluation use only the finalized accepted manifest.

Each model receives one global raw-logit threshold from DEVELOPMENT negatives,
with empirical negative-clip FPR at most 1%. The final Urban FPR may be higher.
This is a fraction of weak negative-label crops, not false alarms per hour.
Positive development labels do not choose that cutoff. All heads, thresholds,
encoder identities and cohort hashes are frozen before the final batch is scored.
The final-use ledger prevents an ordinary rerun from silently consuming the
cohort again. Subsequent tuning makes it development evidence.

Grouped 25%, 50% and 100% learning curves use DEVELOPMENT only, with three
predeclared partial-group selections. Single-group sources remain present at
every size. Their curves measure the available group additions, rather than
establishing a universal required amount of training audio.

## Run and resume

The standard numerical environment is installed with `scripts/setup.sh`.
The locked notebook runtime adds both encoder dependencies:

```bash
uv sync --project notebooks/runtime --frozen --extra cpu --extra probe
```

Use the GPU extra instead of the CPU extra for a deliberately GPU-backed run.
Source acquisition from a public Drive folder requires no Drive mount or account
credentials. The HTML route can change; missing files or an incomplete listing
fail explicitly instead of selecting a small fallback.

```bash
PYTHONPATH=src .venv/bin/python scripts/acquire_public_drive_corpus.py \
  --folder-id PUBLIC_FOLDER_ID --output work/corpus-study-v1
```

Use a new output directory for a different acquisition plan. Transport URLs are
part of acquisition identity: do not change the plan inside an existing partial
download directory. The source archives are never relabeled or overwritten by
the study. A mounted Drive `source-archives` directory can instead be supplied as
the study's `archives` directory when its completed receipts and publisher
checksums agree with the intake manifest.

The UrbanSound metadata CSV must match SHA-256
`b207f7a53f9e5eb02a5dd71d58fb8b66c88c24c0637b926dc72d9d400a6a2354`.
Place the verified CSV at `prepared/metadata/urbansound.csv` and supply the same
file to `--urban-csv`. Fetch model assets with `scripts/fetch_beats.py` and
`scripts/fetch_clap.py`; asset hashes are checked before either encoder loads.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_corpus_milestone.py \
  --study work/corpus-study-v1 \
  --python .venv/bin/python \
  --beats-python notebooks/runtime/.venv/bin/python \
  --clap-python notebooks/runtime/.venv/bin/python \
  --beats-assets work/model-assets/beats \
  --clap-assets work/model-assets/clap \
  --device cpu --feature-shards 2 \
  --urban-csv work/corpus-study-v1/prepared/metadata/urbansound.csv
```

Use `--device cuda --feature-shards 1` with the GPU runtime for explicit GPU
inference. CPU and GPU feature caches must use separate study directories.

Stages leave per-container preparation receipts, content-bound feature caches,
the accepted manifest, fitted numeric heads, development curves, a final-use
ledger and per-case predictions. The runner resumes completed work and refuses
corrupt or incompatible identities. Fresh executions extract each family with
disjoint worker inputs; the recorded CPU execution also ran families concurrently.
GPU execution must be selected explicitly in the extraction runner;
availability of a GPU alone is not evidence that it was used.

Generate the report from saved results without refitting:

```bash
PYTHONPATH=src .venv/bin/python scripts/report_corpus_study.py \
  --study work/corpus-study-v1 --output reports/corpus-study-v1
```

Per-source recall, per-class false positives, paired-domain AUC, the nuisance
control and inspectable errors determine what follows. This study cannot by
itself resolve source/class confounding or certify audible-event reliability.
Those questions require comparable field backgrounds and temporal labels;
positive-only measurement collections remain identified as auxiliary evidence.
