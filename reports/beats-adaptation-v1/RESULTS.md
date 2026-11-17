# BEATs adaptation: completed controls and execution status

Captured at 2026-11-17T12:57:56.581092-05:00 (America/New_York). Execution status: **running**. Completed fine-tuned evaluations: **1/4**.

The full cohort has **30,238 crops**. The declared neural study has **198,348 training-crop visits across 30,864 batches** and 3 complete epochs per arm. An unfinished arm contributes no evaluation row.

These are known-corpus development comparisons, not fresh generalization tests. Each fold excludes its whole held field source from fitting and model selection. Negative training and selection are matched within each fold.

| Encoder | Held source | Positive mix | Positive crops / groups | FSD crop FPR | Field drone recall | Field background FPR |
|---|---|---|---:|---:|---:|---:|
| frozen | wsu-air3 | current | 976 / 4 | 0.92% | 85.13% | 44.74% |
| frozen | wsu-air3 | broader | 1,034 / 5 | 0.77% | 83.92% | 38.95% |
| frozen | audrok-dresden | current | 976 / 4 | 1.22% | 53.45% | — |
| fine-tuned | audrok-dresden | current | 976 / 4 | 1.78% | 58.62% | — |
| frozen | audrok-dresden | broader | 2,462 / 7 | 5.01% | 43.10% | — |

[Runtime-control provenance](MATCHED-RUNTIME.md) distinguishes these matched native-runtime controls from the earlier cached-CPU references.

[Complete paired error analysis: audrok-dresden/current](paired-analysis/audrok-dresden/current/README.md).

The frozen controls show that adding drone-positive sources does not consistently improve transfer. Completed adapted rows, when available, must be compared with their matching frozen row at the same selection rule. Incomplete training, synthetic software checks and training-batch loss are not detector performance results.

## Execution and compute

The supervisor has accumulated 27.43 hours against a 44-hour cumulative budget. Its active child is verified. Recovery checkpoints are configured every 100 steps. A saved 'running' status is treated as interrupted when its matching process is absent.

Actual first-arm training logged 1,675 steps in 3.82 hours (8.22 seconds/batch). Extrapolating to all four arms gives about **70 hours of training alone**, before selection, feature extraction or interruption replay. This supersedes the initial 30-hour estimate from repeated synthetic batches. At the observed rate the full CPU study will not finish inside its 44-hour cap; the cap remains in force.

The most recent recovery logged an exact resume from optimizer step 3,400. Unsaved batches after a checkpoint are replayed; they do not become extra training epochs or independent evidence.

The latest logged training position is epoch 2, batch 704/2,721, optimizer step 3,425.

The native frontend, source folds, three-epoch recipe and calibration rule are unchanged. The best trained epoch is selected on weighted selection-DEV log-loss; a C1 head is then refitted on TRAIN embeddings. Held-field/FSD scores never select that epoch or cutoff.

Positive training spans four, five or seven capture groups. The same 7,931 FSD recordings appear in both folds; they are not 15,862 independent negatives. WSU background windows come from one recording. Publisher presence labels have not received independent crop-audibility review. Crop rates do not establish false alarms/hour or event recall.
