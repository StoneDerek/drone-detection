# BEATs optimization and coverage audit

**The frozen encoder and trained heads already fit TRAIN extremely well.** A
substantial TRAIN-to-selection gap exists before encoder fine-tuning begins. The
first adapted arm's joint-head selection loss is still worse than its frozen
reference after two epochs. The prescribed final encoder selection and C1 head
refit remain pending; these diagnostics do not establish their outcome.

The [complete audit](audit.json) was captured at **2026-11-16 21:04 America/New_York**
(2026-11-17 02:04 UTC). It covers the full 30,238-row cohort, every planned TRAIN
schedule and the complete 4,525-row selection pool. It runs no additional encoder
forward passes and calculates no held-source or FSD scores. The training recipe,
partitions, selection rule, cutoff and cumulative 44-hour cap remain unchanged.

## Frozen TRAIN versus selection

| Held source | Positive mix | Frozen TRAIN weighted log-loss | Frozen selection weighted log-loss |
|---|---|---:|---:|
| AuDroK | Current | 0.000112 | 0.583962 |
| AuDroK | Broader | 0.002668 | 2.385897 |
| WSU | Current | 0.000108 | 0.617624 |
| WSU | Broader | 0.000203 | 0.333566 |

These calculations use the already fitted paired heads and verified native-base
embeddings; no classifier is refitted. Selection weights are fixed across these
comparisons; TRAIN mixtures and weights differ. Log-loss measures confidence as
well as errors, so these values do not replace recall or false-positive rates.

The cached-feature calculation reproduces the first arm's native starting
selection loss within **1.07 × 10⁻⁷**. A large starting preprocessing or frozen-head
arithmetic discrepancy was not found by this check. This does not certify every
possible preprocessing choice or recording label.

## Completed epochs and inspected state

| AuDroK-held/current arm | Joint-head selection weighted log-loss |
|---|---:|
| Frozen starting reference | 0.583962 |
| Full epoch 1, 2,536 steps | 1.921553 |
| Full epoch 2, 5,072 steps | 1.458973 |
| Full epoch 3 | Pending at capture |

The inspected rolling checkpoint had **5,600 optimizer steps**, two completed
epochs and 528 batches completed in epoch 3. Its history agrees with the epoch
log entries. All **977 model/optimizer tensors** inspected were finite. Learning
rates and weight decay matched the recipe; the initial TRAIN scaler was exactly
preserved. Numerical trainer and driver hashes match the running implementation.
The 27 sampled batches logged again during recovery have identical recorded loss
and gradient norm. Replayed batches are not additional independent examples.

The checkpoint SHA identifies the bytes read during the audit. That rolling file
can later be replaced or retired by training; permanent retention of this large
intermediate optimizer file is not promised.

Epoch 2 improved on epoch 1 but remains worse than the frozen reference on this
metric. The predeclared rule still selects the lowest joint-head DEV loss among
epochs 1–3, then refits a C1 head on TRAIN embeddings from the selected encoder.
Interim joint-head loss does not determine the final refitted head's performance.
No unfinished arm enters the completed evaluation comparison.

## Coverage and weighting

All **12 complete planned schedules**—four arms, three epochs each—were rebuilt
from the pinned plan and cohort: **198,348 crop visits in 30,864 batches**. Every
selected crop appears exactly once per epoch, and each batch has one native
waveform length. This verifies complete schedules; sparse runtime logs are not
per-example visit instrumentation.

Class, source-within-class and group-within-source weight allocations match the
protocol. In the active arm, 976 of 16,238 crops are positive (6.01%), receiving
50% of objective weight. Berne's 163 positives in one conservative group receive
25% of total weight; Poland's 813 positives across three groups receive another
25%. This is intentional weighting, not 50/50 sampling or evidence of hundreds
of independent positive captures.

The active arm has 2,536 batches per epoch, including 462 singleton batches from
exact-length bucketing. Batch weight mass varies. The trainer divides weighted
BCE by the fixed epoch-average batch mass rather than each batch's own mass.
At fixed parameters, averaging these batch objectives reproduces the full weighted
objective for every schedule to numerical precision. Parameters change during SGD;
this identity does not certify an optimal sampler, learning rate or trajectory.

## Why printed loss can be zero

`batch_weighted_loss_sum` contains only a weighted BCE numerator. It omits the
batch normalizer and head L2 penalty, and is sampled every 25 steps. It is not
an epoch-average TRAIN loss or full TRAIN accuracy measurement.

The checkpoint's head norm implies a nonzero L2 penalty of about **0.000259** and
a regularization-gradient norm of about **0.000228**. Small nonzero gradient norms
are therefore compatible with a printed zero data-loss numerator.

A separate float32 numerical witness also shows PyTorch BCE reporting exactly
zero for a confidently correct negative logit with a nonzero gradient. Its four
scalar logits and gradients are in the JSON. These are numerical calculations,
not acoustic examples or measurements of unlogged training logits.

## Interpretation and reproduction

The audited failure modes—omitted scheduled crops, unintended weight allocation,
changed scaler or optimizer recipe, nonfinite checkpoint tensors and mismatched
starting validation arithmetic—were not found. This does not establish the cause
of poor transfer. Source-specific cues, limited independent positive diversity,
label ambiguity, calibration and feature adaptation remain alternatives.

The useful next model evidence is the first **completed** adapted arm, including
the final head refit and paired source-level comparison. The public FPV candidate
remains unscored and outside this experiment.

Run from this audit's code revision with the separate Torch runtime while the
first arm's rolling recovery checkpoint still exists:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /path/to/beats-runtime/bin/python scripts/audit_beats_optimization.py \
  --work work/beats-adaptation-cpu2-v1 \
  --output work/optimization-audit-reproduction
```

Plan, manifest, matrix, head, code and observed checkpoint hashes are recorded.
A later run captures later progress. The script uses one checkpoint file descriptor
for checksum verification and safe weights-only loading; atomic replacement cannot
swap the loaded bytes. A mismatched receipt fails explicitly for retry. The script
does not promote receipts or change training state. Public output contains aggregates
and hashes, with no audio paths, recording lists or new learned weights.

Software validation passed **202 tests**, nine optional skips and Ruff. Numerical
and software checks are distinct from drone-performance evaluations.
