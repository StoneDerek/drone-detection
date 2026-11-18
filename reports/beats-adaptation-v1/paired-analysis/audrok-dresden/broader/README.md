# Completed BEATs arm: paired comparison

Captured 2026-11-17T21:52:31.179433-05:00 (America/New_York). Held source: **audrok-dresden**; positive mix: **broader**.

All three epochs completed: **8,163 steps**, selected epoch **2**. The final C1 head is refitted and its DEV cutoff locked. All **7,989** prescribed validation rows are compared; no reduced cohort.

| Source / publisher label | Clips | Frozen errors | Adapted errors | Corrected | New errors | Both wrong |
|---|---:|---:|---:|---:|---:|---:|
| audrok-dresden / drone | 58 | 33 | 22 | 11 | 0 | 22 |
| fsd50k-eval / non_drone | 7,931 | 397 | 301 | 261 | 165 | 136 |

Positive-label errors are missed drone clips; negative-label errors are false-positive clips. Each classifier uses its own cutoff from the same prescribed DEV rule. The paired counts are descriptive, with no independent-clip significance claim.

## What the final head refit changed

| Classifier stage | TRAIN weighted log-loss | Selection-DEV weighted log-loss |
|---|---:|---:|
| frozen | 0.002668 | 2.385897 |
| selected_joint_head | 0.017347 | 0.929070 |
| refitted_C1_head | 0.000292 | 0.810686 |

The selected encoder and saved joint head reproduce their recorded DEV loss within 1.23e-07. This separates the interim joint-head diagnostic from the final refitted classifier. Log-loss measures confidence as well as classification errors.

## Evidence and limits

[paired-comparison.json](paired-comparison.json) retains complete source-level evaluation receipts, paired counts, cutoffs and input/model/code hashes. Reconstructed confusion counts and DEV cutoff calculations exactly match the locked completed artifacts. The complete per-record paired ledger remains private.

- Publisher labels are weak and crop audibility has not been independently reviewed.
- Correlated excerpts/capture groups prevent treating clip counts as independent field trials.
- FSD false-positive clip rates do not establish false alarms/hour.
- This single completed arm does not settle the four-arm study or identify the cause of transfer failures.

The remaining study arms continue under the original recipe and cumulative 44-hour budget. Public FPV candidates remain outside this experiment.
