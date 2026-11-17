# Completed BEATs arm: paired comparison

Captured 2026-11-17T01:19:49.380072-05:00 (America/New_York). Held source: **audrok-dresden**; positive mix: **current**.

All three epochs completed: **7,608 steps**, selected epoch **3**. The final C1 head is refitted and its DEV cutoff locked. All **7,989** prescribed validation rows are compared; no reduced cohort.

| Source / publisher label | Clips | Frozen errors | Adapted errors | Corrected | New errors | Both wrong |
|---|---:|---:|---:|---:|---:|---:|
| audrok-dresden / drone | 58 | 27 | 24 | 3 | 0 | 24 |
| fsd50k-eval / non_drone | 7,931 | 97 | 141 | 40 | 84 | 57 |

Positive-label errors are missed drone clips; negative-label errors are false-positive clips. Each classifier uses its own cutoff from the same prescribed DEV rule. The paired counts are descriptive, with no independent-clip significance claim.

## What the final head refit changed

| Classifier stage | TRAIN weighted log-loss | Selection-DEV weighted log-loss |
|---|---:|---:|
| frozen | 0.000112 | 0.583962 |
| selected_joint_head | 0.000056 | 0.823175 |
| refitted_C1_head | 0.000011 | 0.588679 |

The selected encoder and saved joint head reproduce their recorded DEV loss within 3.18e-08. This separates the interim joint-head diagnostic from the final refitted classifier. Log-loss measures confidence as well as classification errors.

## Evidence and limits

[paired-comparison.json](paired-comparison.json) retains complete source-level evaluation receipts, paired counts, cutoffs and input/model/code hashes. Reconstructed confusion counts and DEV cutoff calculations exactly match the locked completed artifacts. The complete per-record paired ledger remains private.

- Publisher labels are weak and crop audibility has not been independently reviewed.
- Correlated excerpts/capture groups prevent treating clip counts as independent field trials.
- FSD false-positive clip rates do not establish false alarms/hour.
- This single completed arm does not settle the four-arm study or identify the cause of transfer failures.

The remaining study arms continue under the original recipe and cumulative 44-hour budget. Public FPV candidates remain outside this experiment.
