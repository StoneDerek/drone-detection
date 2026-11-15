# Frozen audio comparison

This development workflow compares mean-pooled frozen BEATs or CLAP features
with a regularized linear classifier. It can run while the new corpus downloads.
The current real-audio verification uses only five retained browser derivatives
(three training, two development recordings). It establishes software operation,
not improved generalization or a useful detector. Existing deployed models and
the broader data-first study remain unchanged.

## Setup and verified local assets

From the repository root, create the separate audio-model runtime. This keeps
Torch/Transformers out of the ordinary inference environment. `cpu` and `gpu` are
mutually exclusive; use `gpu` on an allocated compatible CUDA host.

```bash
UV_CACHE_DIR=work/uv-cache uv sync --project notebooks/runtime --frozen --extra cpu --extra probe
PYTHONPATH=src notebooks/runtime/.venv/bin/python scripts/fetch_beats.py --output work/beats-encoder
PYTHONPATH=src notebooks/runtime/.venv/bin/python scripts/fetch_clap.py --output work/clap-encoder
```

Fetch scripts download explicit pinned assets once, check sizes and SHA-256, and
reuse only verified complete files. Inference uses local assets without network
access. The BEATs official OneDrive download was unavailable; the chosen named
checkpoint comes from an immutable third-party mirror whose bytes are pinned,
but were not independently compared with the official download. This provenance
limit is retained in the asset manifest. CLAP uses an immutable official LAION
Hugging Face release. Keep the supplied model/source terms and asset manifests.

## Reproduce the retained-example software check

Before using the incoming corpus, prepare the five already-published browser
excerpts. This command checks their pinned derivative hashes, retains their
original parent identities/splits and weak recording labels, and writes a
software-only manifest. It downloads no new audio.

```bash
PYTHONPATH=src notebooks/runtime/.venv/bin/python scripts/prepare_frozen_probe_smoke.py \
  --output work/frozen-probe-smoke
```

Then use `work/frozen-probe-smoke/manifest.jsonl` in the comparison command below.
Three related Svanstrom recordings are TRAIN and two Batear recordings are
DEVELOPMENT. This is not a statistical model-selection or generalization study.

## One-command comparison

Supply an existing schema1 JSONL manifest with reviewed provenance and explicit
`train`/`development` assignments. Both splits require both recording classes.
Related recordings and duplicate components cannot cross splits. Unknown and
uncertain labels are excluded. Metadata for `final` rows is validated, but their
audio is never opened by this runner; there is intentionally no final-test flag.

```bash
PYTHONPATH=src notebooks/runtime/.venv/bin/python scripts/run_frozen_probe.py \
  --manifest /path/to/reviewed-manifest.jsonl \
  --encoder-family beats --encoder work/beats-encoder \
  --output work/comparison/beats --cache work/features/beats \
  --reference-bundle artifacts/real-v2/baseline
```

For CLAP, use `--encoder-family clap --encoder work/clap-encoder` and separate
output/cache directories. The ordinary relative-Mel baseline needs no TensorFlow.
Optional existing CNN/YAMNet references require their declared TensorFlow runtime
as well. A reference must accept the same canonical sample rate and context.
Its weights remain fixed; a recording threshold is selected on development data.

Use `--device cuda` explicitly for a CUDA run. CPU defaults to two Torch threads;
`--threads` sets and records this choice. Start with the default 3-second context,
0.5-second hop, 0.25-second minimum, and at most 60 uniformly selected contexts
per recording. Timing and channel settings are fixed in the run plan. A small
throughput pilot should precede a whole-corpus run.

## Representation, labels, and decisions

- Decode the declared physical `channel_index` (default 0), preserve amplitudes,
  and resample offline to 16 kHz. No stereo/array averaging is applied.
- Include short eligible contexts using explicit right-zero-padding. Do not
  exclude quiet examples or silently label every window from a positive file.
- BEATs uses its released 128-band Kaldi frontend, normalization, and frozen
  768-dimensional time-frequency token mean. CLAP resamples the canonical16 kHz
  signal to its native 48 kHz frontend and uses the released repeat-padding policy
  to 10 seconds, producing a normalized 512-dimensional projection. Upsampling
  does not recover frequencies lost in the canonical16 kHz signal. These are
  different complete pretrained pipelines, not an isolated architecture ablation.
- Fit one recording label to the **mean of selected context embeddings**. Train
  the scaler and logistic head using TRAIN only. Each group contributes equal
  mass within its class; class masses are balanced. `C=1` and seed42 are fixed
  defaults, not an architecture/head search.
- Recording score is `sigmoid(mean context logits)`, which equals scoring the
  mean embedding for this linear head. It is not the mean of probabilities or
  an any-hit detector. A mean-pooled positive recording is a weak supervision
  baseline and may dilute brief drone evidence.
- Select a global recording threshold on a fixed development grid by macro
  source balanced accuracy, then negative recording FPR and deterministic ties.
  This recording-level rule is not the operational false-alarms/hour contract.
- Reuse existing recording confusion and grouped-bootstrap metric routines.
  Report each source separately; single-class sources have undefined balanced
  accuracy/AUC. Whole weak-labelled recording hours are not verified audible
  drone hours or drone-absent exposure. Two development examples cannot support
  a statistical or field-reliability conclusion.

The uniform cap bounds extraction work and includes endpoints when the cap is
at least2. A capped trace is sampled diagnostic output, not continuous monitoring.
Window scores have no per-window ground truth. Event recall, false alarms/hour,
alarm occupancy, and detection delay require a separate temporal evaluation with
sufficient annotations and complete causal replay.

## Outputs and recovery

Each run writes:

- `plan.json`: manifest/code/encoder identities, settings, supervision and split rules.
- `head/model.json` and `head/parameters.npz`: numeric parameters and identities;
  no pickled classifier. Supply the separately verified encoder to reload it.
- `comparison.json`: development-only results, selected thresholds and trials,
  source breakdowns, predictions, and optional fixed-reference comparison.
- `traces/*.jsonl`: exact selected context timestamps, padding, logits and scores.

Each completed feature-cache record has an atomic completion receipt and payload
hash. Rerunning the same command reuses completed entries and refits/rechecks the
small deterministic head. An interruption within a recording recomputes that
recording; partial payloads are not accepted as complete. Source changes, corrupt
completed caches, or mismatched run identities fail explicitly. Use a new output
for changed code/configuration/manifest/encoder/device. One writer per run/cache
is required. Local recovery is not verification of Google Drive synchronization.

## Incoming original-source preparation

`scripts/prepare_comparison_manifest.py` accepts one JSON source plan. Paths are
relative to that plan file. A plan can combine the two TUT2017 releases,
UrbanSound8K, DNC and explicit per-file mappings for other collections:

```json
{
  "sources": [
    {
      "adapter": "tut2017",
      "release": "evaluation",
      "metadata_root": "TUT-acoustic-scenes-2017-evaluation",
      "audio_root": "TUT-acoustic-scenes-2017-evaluation",
      "channel_index": 0
    },
    {
      "adapter": "urbansound8k",
      "metadata": "UrbanSound8K/metadata/UrbanSound8K.csv",
      "audio_root": "UrbanSound8K"
    },
    {
      "adapter": "dnc",
      "audio_root": "DNC"
    }
  ]
}
```

The TUT metadata directory must include `meta.txt`; renamed evaluation filenames
also require `evaluation_setup/map.txt`. UrbanSound paths follow the official
`audio/foldN/name.wav` layout. DNC recursively inventories extracted audio files.

Metadata can be checked before the corresponding audio finishes downloading:

```bash
.venv/bin/python scripts/prepare_comparison_manifest.py \
  --plan intake-plan.json --output work/intake-catalog.jsonl --catalog-only
```

This produces a catalog, **not** a trainable manifest. Full preparation hashes
whole native files, checks native frame/rate/channel/subtype information and
verifies supplied excerpt durations against headers:

```bash
.venv/bin/python scripts/prepare_comparison_manifest.py \
  --plan intake-plan.json --output work/intake-manifest.jsonl
```

Source bytes remain unchanged. Mono/stereo inputs select physical channel 0 by
default; source plans or generic rows can declare another `channel_index`.
Arrays with more than two channels require an explicit valid physical channel.
The frozen-probe runner selects that channel at decoding; legacy decoders may
average channels and should not be substituted silently.

All rows remain `unassigned` with unresolved provenance review by default.
Publisher scene/event classes are auxiliary weak `non_drone` labels, never
verified drone-absent time. TUT original recording IDs and source offsets are
preserved; UrbanSound Freesound IDs/offsets and publisher folds are preserved.
Publisher folds do not override shared parent identity. DNC lacks recovered
capture sessions, so the full source stays in one conservative component.

To assign a reviewed manifest, supply `--review review.json`, an explicit map:

```json
{
  "recording-id-from-catalog": {
    "split": "train",
    "provenance_review": "Reference to the completed source/overlap review"
  }
}
```

Review every member of a component together. Assignment cannot separate related
parents, derivatives, matching hashes or explicit cross-source provenance keys.
The review does not strengthen weak acoustic labels.

Other collections use a `generic` plan entry with `source`, `source_version`,
`metadata` (JSONL) and `audio_root`. Every metadata row needs `path`, an explicit
`label` and `label_evidence`. Preserve known `original_id`, `session_id`,
`source_interval_seconds`, `channel_index`, `events`/`ambiguous` and annotation
strength. A speech filename never implies drone absence. Add shared identities
such as `"provenance_keys": ["freesound:123"]` to link related rows imported in
the same plan. Previously prepared corpora are not automatically searched for
unrecorded cross-source relationships.

Verified here: all six small source-metadata/doc downloads matched the earlier
pinned SHA-256 identities. Actual metadata parsed 4,680 TUT development rows
(248 parent recordings), 1,620 evaluation rows (109 parents), and 8,732
UrbanSound rows (1,297 Freesound parents). The CLI created a 15,032-row catalog
without audio. These are metadata identities and counts, not independent-session
counts or accepted positive/negative hours. Evidence is in
`work/intake-adapters/metadata-verification.json`.

Not completed here: bulk Drive archive verification, native intake of those
collections, detailed source-specific Berne/Polish/NEAPTIDE adapters, or creation
of a new training/evaluation split.
