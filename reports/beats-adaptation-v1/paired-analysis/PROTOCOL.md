# Analysis of completed paired BEATs arms

The analyzer compares the complete validation pool of an adapted arm with its
matched frozen control. It requires all three training epochs, the final C1 head
refit, locked DEV cutoffs and completed evaluation receipts. It rejects unfinished
arms. This analysis plan is recorded before inspecting the first adapted arm's
held-source results; epoch-selection DEV losses are already known.

For the AuDroK-held/current-positive arm, use every prescribed validation row:
**7,931 FSD50K negative crops and 58 AuDroK weak-positive crops**. The 58 positive
crops describe one conservative campaign group, not 58 independent field trials.
The FSD recordings describe 2,122 publisher-uploader groups. Other arms retain
their own complete plan-defined cohorts; there is no subsample fallback.

Each fitted model retains its own cutoff from the existing rule: the maximum
cutoff required for at most 1% empirical FPR on each selection-DEV negative source.
Neither a shared arbitrary probability threshold nor a held-source-derived cutoff
is substituted. Input matrix identities, classifier hashes and model identities
must agree with the completed artifacts. Reconstructed source confusion counts
and DEV cutoff calculations must reproduce their recorded values exactly.

Report paired decisions separately by source and publisher class: both correct,
both wrong, errors corrected by adaptation, and new errors after adaptation. Also
retain the complete confusion counts and cohort/group counts. These are descriptive
comparisons; no independent-clip significance test or field-campaign confidence
interval is inferred. FSD false-positive clip rates are not false alarms/hour.

To interpret the interim joint-head diagnostics, calculate TRAIN and selection-DEV
weighted log-loss for three stages using the completed cached representations:
the frozen classifier, the selected epoch's saved joint training head, and the final
refitted C1 head. The selected encoder plus joint head must reproduce its recorded
selection loss within 1e-4. Held-source decisions use only the two completed final
classifiers; the joint head is not given a newly tuned held-source threshold.

No additional encoder inference or fitting is needed. This analysis does not
change the running experiment, choose a new epoch, alter its data mixture or extend
the cumulative 44-hour budget. Publisher labels remain weak, and prior corpus
exposure prevents calling this a fresh confirmation study. The separately audited
public FPV source remains unscored and outside the experiment.

The full per-record paired ledger stays in ignored private evidence storage. Public
reports contain aggregate comparisons, completed evaluation receipts, model/input
hashes and the ledger checksum. Software tests use binary decision fixtures and
must not be described as drone-performance results.

For a completed arm, run with the separate Torch runtime:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /path/to/beats-runtime/bin/python scripts/compare_completed_beats_arm.py \
  --work work/beats-adaptation-cpu2-v1 \
  --fold audrok-dresden --positive-mix current \
  --output reports/beats-adaptation-v1/paired-analysis/audrok-dresden/current \
  --private-output work/beats-paired-analysis-v1/audrok-dresden/current
```

At the time of writing this plan, neural training for that arm is complete and
selected-encoder feature extraction is still running. No adapted held-source
performance is claimed by this protocol document.
