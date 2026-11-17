# Automatic analysis of completed BEATs arms

The running CPU experiment continues from its existing published numerical
implementation and 44-hour cumulative budget. A separate observer prepares each
completed arm's paired comparison and updated result table. Its outputs stay
local until reviewed and published. It does not send notifications.

The observer verifies the declared cohort, matched frozen control, all three
training epochs, selected encoder checksum, final fitted head, locked selection
cutoff and completed evaluation receipt. It then runs the existing
[predeclared paired analysis](paired-analysis/PROTOCOL.md) on **every** prescribed
validation row. That analyzer reproduces the recorded confusion counts, selection
cutoffs and selected joint-head loss using completed cached representations.
It performs no encoder inference or new fitting. Unfinished arms produce no
comparison row.

An existing completed paired analysis is reused only when its models, cohort,
code, matrices and private ledger agree. Changed completed evidence stops the
observer for investigation. The observer checks every 30 seconds, runs at lower
CPU scheduling priority and uses one BLAS thread. The original trainer remains
the only writer in the training study directory. Observer output paths must be
outside that directory, with private ledgers separate from aggregate reports.

The observer stops when the study completes, pauses or fails. An absent training
child must be confirmed on two successive polls before it stops for interruption;
this allows brief gaps between supervisor stages. It never restarts training or
extends the budget. It holds an exclusive per-study observer lock, independent
of the trainer's writer lock. Its state receipt and logs identify observer
failures without changing training state.

## Local execution

Run from a checkout pinned to the published observer implementation. Initialize a
separate report directory from the current public report so that existing
analysis and interpretation files are retained:

```bash
mkdir -p work/beats-completion-observer-v1
cp -a reports/beats-adaptation-v1 work/beats-completion-observer-v1/report

PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /path/to/beats-runtime/bin/python scripts/watch_completed_beats_arms.py \
  --work work/beats-adaptation-cpu2-v1 \
  --report work/beats-completion-observer-v1/report \
  --private work/beats-paired-analysis-v1 \
  --state work/beats-completion-observer-v1/state.json \
  --python /path/to/beats-runtime/bin/python
```

Use `--once` for a single verification pass. The private output argument points
to the existing first-arm ledger and retains each future arm's full paired
ledger. The local report directory receives aggregate comparisons and fitted
head exports. Audio, encoder weights and per-record ledgers are not published
by the observer. Its background process must still be checked after an
execution-environment replacement; it is not a remote scheduler.

## Verification

The initial real-study pass verified and reused the complete **7,989-row** first
comparison. The selected model, final classifier, original result receipts and
private ledger retained their hashes and modification times. See
[the verification receipt](observer-verification.json).

The [step-4,900 checkpoint audit](checkpoint-integrity-20261117.json) verified its
checksum, finite model/optimizer tensors, unchanged numerical source identities
and unchanged plan. This is a recovery integrity check, not detector performance.
The broader-positive arm is still incomplete.

**34 focused software checks** passed, including incomplete-result exclusion,
reuse of existing analysis, rejection of changed evidence and stopping on
paused/failed/interrupted studies without modifying their budget or restarting
training. The project-scoped Ruff and whitespace checks passed. Small control-flow
fixtures are software tests; they provide no acoustic performance evidence.
