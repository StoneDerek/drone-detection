# Acoustic drone detection experiments

A runnable CPU baseline, controlled improvement experiments, full-recording
evaluation, and an audio replay demo. **Real drone detection reliability is not
established:** the supplied evidence packet contains no audio or model artifacts.
The current model is trained on generated signals for software validation only.

See [results](reports/synthetic-v1/RESULTS.md), the
[evaluation contract](contracts/evaluation-v1.json),
[data conventions](data/README.md), [protocol](docs/protocol.md), and
[evidence/research notes](docs/evidence-and-research.md).

## Install and validate

Use the existing checkout in the task's isolated environment; no worktree is
needed. Python 3.12, `uv` (tested 0.13.5), and libsndfile support are required.
The locked SoundFile wheel supplies libsndfile on the tested Linux x86-64 host.
No GPU, credentials, external services or pretrained weight downloads are needed.

```bash
cd /workspace/drone-detection
bash scripts/setup.sh
bash scripts/check.sh
```

The lockfile pins runtime and test dependencies. Set `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1` for comparable CPU timings; the CLI also limits numerical
thread pools to one worker. Training is deliberately inexpensive.

## Reproduce the controlled experiment

```bash
.venv/bin/drone-detector experiment \
  --output runs/reproduce-v1 --contract contracts/evaluation-v1.json
```

Use a fresh output directory each time; evidence is never overwritten. The run
generates 90 synthetic 12-second recordings, audits and groups them, reserves
generator 2 as final, fits seven controlled arms, tunes on development, and
evaluates the frozen baseline/candidate together once. It creates:

- `corpus/manifest-v1.jsonl`, audit, split definitions, annotations, final-use ledger;
- `experiment-plan.json`, `experiment-ledger.json`, `frozen-selection.json`;
- complete JSON/NPZ model packages, all threshold trials, per-window traces;
- `RESULTS.md`, source breakdowns/counts/intervals in `results.json`;
- `demo.html`: self-contained audio, timestamped scores, alarm state, ground-truth
  spans, model ID, fixed positive/ordinary/hard-negative inputs, and a labeled
  post-hoc failure appendix. Open the HTML as a local file. It is recorded replay,
  not live microphone input or loudspeaker playback.

The saved [reference evidence](reports/synthetic-v1/RESULTS.md) reports 9/10 events
for both baseline and candidate; one versus zero false alarms in only 0.0806
negative hours. Both fail the provisional recall-confidence, false-alarm-bound
and delay targets. Only replay runtime meets its target on the measured CPU.
There is no validated microphone, distance, SNR or environment range.

## File inference and paced recorded replay

The bundled artifact deliberately carries the synthetic limitation:

```bash
.venv/bin/drone-detector infer \
  --bundle artifacts/synthetic-v1/candidate \
  --audio runs/reproduce-v1/corpus/audio/g2-s00-positive.wav \
  --output runs/file-scores.jsonl

.venv/bin/drone-detector stream --realtime \
  --bundle artifacts/synthetic-v1/candidate \
  --audio runs/reproduce-v1/corpus/audio/g2-s00-positive.wav \
  --output runs/replay-scores.jsonl
```

`stream --realtime` prints scores as 100 ms audio blocks arrive at wall-clock
pace. Both commands use identical context/score/alarm logic; runtime metadata
is saved alongside each trace. File decoding/resampling precedes replay. The
engine accepts canonical 16 kHz mono blocks; live resampling/device latency is
unimplemented and unmeasured. Short inputs explicitly pad or return insufficient
audio; silence never generates an alarm. Scores are not calibrated probabilities.

The fixed-set HTML can also be regenerated from the reference package:

```bash
.venv/bin/drone-detector demo \
  --bundle artifacts/synthetic-v1/candidate \
  --manifest runs/reproduce-v1/corpus/manifest-v1.jsonl \
  --selection reports/synthetic-v1/demo-selection.json \
  --output runs/reference-demo.html
```

The manifest hash is checked; if generation/annotations change, create and freeze
a new model/protocol rather than substituting unrelated data.

## Authentic data workflow (awaiting audio)

Obtain `Drone Detection Project-20261108T015626Z-1-001.zip` and provenance/permission
for its audio. Complete original/session mapping and temporal annotations first.
Follow the [collection and labeling protocol](docs/collection-protocol.md).
The historical file inventory is not an audio dataset or verified class map.

```bash
.venv/bin/drone-detector audit --root /path/to/audio \
  --annotations /path/to/annotations.jsonl --output runs/authentic/inventory-v1.jsonl
.venv/bin/drone-detector split --manifest runs/authentic/inventory-v1.jsonl \
  --output runs/authentic/manifest-v1.jsonl --seed 42
.venv/bin/drone-detector train --manifest runs/authentic/manifest-v1.jsonl \
  --representation relative_mel --output runs/authentic/trained
.venv/bin/drone-detector tune --bundle runs/authentic/trained \
  --manifest runs/authentic/manifest-v1.jsonl --contract contracts/evaluation-v1.json \
  --output runs/authentic/frozen
.venv/bin/drone-detector evaluate --bundle runs/authentic/frozen \
  --manifest runs/authentic/manifest-v1.jsonl --split development \
  --output runs/authentic/development
```

Use `split --final-source NAME` to reserve a whole source where feasible. Resolve
duplicate/excerpt review before splitting; uncertain provenance groups an entire
source. Train and tune refuse weak/unverified supervision and cross-split
derivatives. Inspect development failures and freeze the complete comparison
before final testing. The command below is the **one-shot final step**, not part
of an iterative development loop:

```bash
.venv/bin/drone-detector evaluate --bundle runs/authentic/frozen \
  --manifest runs/authentic/manifest-v1.jsonl --split final \
  --output runs/authentic/final
```

Pass every prespecified comparison package after `--bundle` in that one call.
Final-use history is retained beside the manifest. `--allow-final-reuse` explicitly
labels repeated final assessment as development evidence. It cannot restore an
untouched holdout. No authentic training/evaluation command above has been run
because the required audio is missing.

## Portable model package

[Artifact notes](artifacts/synthetic-v1/README.md) identify weights, training-only
statistics, dependencies, source snapshot and hashes. Inference uses JSON and
NPZ with `allow_pickle=False`; omitted or tampered statistics cause an error.
The wheel and package were tested in a second isolated virtual environment from
an unrelated working directory, with no editable import or notebook state.
See [validation evidence](reports/synthetic-v1/VALIDATION.md).
