# DADS: complete path and label metadata census

All **180,320 rows across 39 Parquet shards** were inspected at pinned revision
`981b832c35b45a57518c989f0f79101adb4ae91f` of
[Drone Audio Detection Samples](https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples).

**Every stored path is a class-and-sequence filename.** The release supplies
`no-drone-1.wav` through `no-drone-16729.wav` and `drone-1.wav` through
`drone-163591.wav`. These are unique clip identifiers; they do not provide an
original source, recording, session, microphone or crop offset.

| Complete metadata finding | Count |
|---|---:|
| Negative-label rows | 16,729 |
| Positive-label rows | 163,591 |
| Unique literal paths | 180,320 |
| Class-and-sequence paths | 180,320 |
| Filename class consistent with label | 180,320 |
| Null paths | 0 |

All 39 physical schemas contain only `audio.bytes`, `audio.path` and `label`.
The audit projected **only path and label**; it extracted or decoded no audio
member and generated no classifier score. Whole-shard LFS checksums were not
verified. Range-cache blocks can contain incidental compressed audio bytes, so
metadata projection should not be described as zero audio-related network bytes.
The retained cache contains 123,268,619 bytes of unique range payload; this is not
a measurement of total network traffic including retries.

## What this means for the project

The dataset card lists six positive source collections: Al-Emadi's Drone Audio
Dataset, SPCup19 Egonoise, DREGON, DroneNoise, AUDROK, and the drone fault-classification
dataset. Its negative-source bibliography lists UrbanSound8K, TUT2017 evaluation,
ESC-50 and DNC. Source names overlap collections already used in this project.
The bibliography does not map individual DADS clips to those collections.

Consequently, a filename-only audit of this export cannot recover original
recording groups. The large number of unique filenames does not establish a
large number of independent captures. This finding does not establish actual
waveform duplication, train/test leakage, acoustic label correctness or the cause
of the project's generalization failures. It also does not make the audio useless
for training.

For positive-data expansion, the useful next step is to inspect the original
source distributions and retain their recording provenance. If the aggregate is
used, parent relationships and cross-project overlap need separate evidence;
waveform matching or additional publisher metadata might recover some links.
Keep related recordings within one partition. A random clip split of this export
does not by itself supply an independent-recording evaluation.

Acquisition context also needs checking upstream: titles involving ego-noise or
fault classification do not automatically establish relevance to a microphone
listening for an unfamiliar drone in the field. No new training or confirmation
role is assigned solely from those titles or this filename census.

## Reproduction and evidence

[census.json](census.json) records complete counts, all shard receipts, projected
metadata hashes and method identities. [method.json](method.json) separates the
verified metadata findings from unverified waveform properties. The full per-row
path/label tables and cached source metadata remain in ignored local storage.

Use PyArrow **22.0.0** in a separate audit runtime. For example, with the project's
Python environment available:

```bash
UV_CACHE_DIR=work/dads-source-census-v1/uv-cache uv pip install \
  --python .venv/bin/python --target work/dads-source-census-v1/runtime \
  --no-deps pyarrow==22.0.0
```

Cache the publisher metadata as `dataset.json`, `data-tree.json` and `README.md`
under `work/dads-source-census-v1`, using these pinned endpoints:

- `https://huggingface.co/api/datasets/geronimobasso/drone-audio-detection-samples/revision/981b832c35b45a57518c989f0f79101adb4ae91f`
- `https://huggingface.co/api/datasets/geronimobasso/drone-audio-detection-samples/tree/981b832c35b45a57518c989f0f79101adb4ae91f/data?recursive=false`
- `https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples/raw/981b832c35b45a57518c989f0f79101adb4ae91f/README.md`

Then run:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/audit_dads_metadata.py \
  --work work/dads-source-census-v1 --workers 4
```

The script requires the pinned complete release, checks every projected schema,
and rejects incomplete row/label totals. It uses exact cached HTTP ranges and
preserves publisher LFS hashes without pretending partial reads verify those
whole-file hashes. Two shards initially encountered temporary HTTP 503 errors;
one cache-reusing retry at lower concurrency completed all 39. The published
method was then rerun across the complete cached cohort and all physical schemas.
Waveform durations, channel/rate format, acoustic content, duplication and original
capture independence remain unverified by this metadata-only audit.

The running BEATs data, models, selection rule and cumulative 44-hour budget are
unchanged. No DADS waveform was added to or scored by that experiment.
