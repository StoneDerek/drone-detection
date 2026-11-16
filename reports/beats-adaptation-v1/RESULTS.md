# Positive-diversity development controls

The four frozen-encoder controls are complete. Full BEATs fine-tuning is a separate running/queued phase; the results below do not claim that the encoder has been task-trained.

These are **known-corpus development comparisons**, not fresh generalization tests. Each fold excludes its entire field source from fitting and model selection. Negative training and selection are identical across the two positive mixtures within a fold.

| Held field source | Positive TRAIN mix | Positive crops / groups | FSD crop FPR | Held-field drone recall | Held-field background FPR |
|---|---|---:|---:|---:|---:|
| wsu-air3 | current | 976 / 4 | 0.92% | 85.13% | 44.74% |
| wsu-air3 | broader | 1,034 / 5 | 0.77% | 83.92% | 38.95% |
| audrok-dresden | current | 976 / 4 | 1.24% | 53.45% | — |
| audrok-dresden | broader | 2,462 / 7 | 5.01% | 43.10% | — |

Broader positive training does not consistently help in this frozen representation. Adding AuDroK positives modestly reduces WSU background errors, with a small recall loss. Adding WSU positives lowers AuDroK recall and increases FSD false positives. This does not establish whether weak labels, source coverage, learned features or another factor causes the failures.

For the AuDroK-held fold, both mixtures include 190 WSU negative windows in TRAIN. The broader mixture also includes WSU positives. For the WSU-held fold, AuDroK contributes only additional positives. WSU background and positive crops from the same capture are never split between fitting and held-field validation.

The current positive TRAIN has 976 crops from Poland and Berne, spanning four groups. The broader mixtures add either 58 AuDroK crops from one campaign, or 1,486 WSU positive windows from three date/site groups. The 7,931 FSD recordings are reused across the two folds; they are not 15,862 independent negatives. WSU's 190 negative windows come from one recording. Labels remain weak publisher labels; audibility review is pending.

## Full encoder comparison

The next four arms use all 90,311,792 BEATs representation parameters, three complete epochs, batch size 8, and the unchanged native frontend. Each adapted model starts from its paired trained C1 head. The best trained epoch is selected using weighted selection-DEV log-loss, then the same C1 head is refitted on TRAIN embeddings and the same global calibration rule is applied. No held-field or FSD score selects the checkpoint or cutoff.

Training recovery stores model, optimizer, CPU/CUDA/Python/NumPy RNG and the exact next batch. Checksums and pending receipts protect interrupted checkpoint publication. Every selected training crop is visited once per epoch; exact-length batches avoid adding waveform padding. A 44-hour cumulative pipeline budget requests a cooperative checkpoint pause rather than reducing the dataset or epoch count.

The GPU notebook reconstructs the full corpus from existing pinned originals and public field sources. It creates its own base embeddings and does not require a previous feature cache. Canonical audio and checkpoints persist in a separate private Drive project; verified local copies feed training.

The existing blind field-review inventory remains useful while training runs. Full-backbone updates and training/recovery mechanisms have been checked on synthetic signals; those software checks are not detector performance evidence.
