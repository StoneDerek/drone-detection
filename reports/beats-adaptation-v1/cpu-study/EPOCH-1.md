# First full-epoch diagnostic

This is the epoch-1 snapshot. The later [optimization audit](../optimization-audit/README.md)
adds epoch 2 and complete-schedule checks without changing the study recipe.

The AuDroK-held/current-positive arm completed its first full epoch: all 16,238
selected training crops, 2,536 optimizer steps. The prescribed recipe has three
epochs. This is an interim development diagnostic, not a completed adapted model
evaluation.

| Selection-DEV diagnostic | Weighted log-loss |
|---|---:|
| Initial frozen encoder and trained C1 head | 0.583962 |
| Encoder and joint training head after epoch 1 | 1.921553 |

Lower is better, so the initial joint-head validation result has deteriorated.
That result remains in the record. It does not by itself establish whether the
selected encoder's representation will be worse after the final C1 head is
refitted, or how either model will transfer to the held source.

Two prescribed epochs remain. Selection is still the lowest weighted DEV loss
among epochs 1–3; the final C1 classifier/scaler refit and held-source evaluation
follow the original protocol. No learning rate, data composition, checkpoint
selection rule or decision threshold has been changed in response to this
diagnostic. No held-source score was inspected.

[epoch-1-progress.json](epoch-1-progress.json) records the verified encoder
checkpoint hash and scope. The separately identified public source candidates
remain unscored and outside this training/selection cohort.
