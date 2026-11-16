# Positive coverage versus encoder adaptation

This is a development study on the complete previously consumed corpus. It
tests whether adding public drone-positive sources, adapting BEATs, or combining
the two improves transfer beyond sources used for fitting. It does not provide
a new independent test or a near-human performance claim.

Two whole-source validation folds hold out WSU or AuDroK from both fitting and
selection. FSD remains validation-only in both folds. The original TRAIN and
selection pools come from the preceding confuser study.

| Held source | Positive mix | TRAIN crops | Positive crops / groups | Selection crops | Validation crops |
|---|---|---:|---:|---:|---:|
| WSU | Current | 16,048 | 976 / 4 | 4,525 | 9,607 |
| WSU | Broader: add AuDroK | 16,106 | 1,034 / 5 | 4,525 | 9,607 |
| AuDroK | Current | 16,238 | 976 / 4 | 4,525 | 7,989 |
| AuDroK | Broader: add WSU | 17,724 | 2,462 / 7 | 4,525 | 7,989 |

In the AuDroK-held fold, both positive mixtures include the 190 WSU negative
windows in TRAIN. Negative training is matched within each fold. Adding WSU
positives while keeping their same-recording negatives in validation would leak
capture information; this protocol keeps those groups intact.

Every mixture has a frozen-encoder control and a fully adapted BEATs arm. The
native filterbank, mean token pooling, physical channel, canonical crops and
sample rate stay fixed. There is no new architecture, augmentation or frontend
sweep. Weights give equal mass to classes, sources within class and capture
groups within source. The final classifier is C=1 logistic regression with
weighted TRAIN-only scaling and total sample-weight mass 9,959.

Adapted arms initialize from the paired trained head. All 90,311,792 encoder
parameters become trainable. The recipe is three complete epochs, batch size 8,
seed 42, float32, AdamW, encoder learning rate 1e-5, head learning rate 1e-4,
encoder weight decay 0.01, C1 head regularization and gradient clipping at norm
1. Exact-length batches visit every selected crop once per epoch without
waveform padding. Different mixture sizes imply different optimizer-step counts
under this shared epoch policy; those counts are recorded.

The lowest weighted selection log-loss among trained epochs 1–3 selects the
encoder. Epoch 0 is a reference, not an adapted result. The final C1 head and
scaler are refitted on selected-encoder TRAIN embeddings. One global cutoff is
the maximum of the thresholds needed for at most 1% empirical FPR on each
selection negative source. Held-source and FSD scores do not select an epoch,
head, cutoff or learning-rate setting.

Checkpoints preserve model, optimizer, Python/NumPy/CPU/CUDA RNG and the exact
next batch. Data, source, runtime and parameter identities are checked on reuse.
The CPU continuation uses 100-step recovery checkpoints; GPU defaults to 1,000
steps. An explicit cadence-only migration verifies the full non-identity state
fingerprint while keeping the numerical trainer and driver unchanged.
One notebook/runtime may write a project at a time. A 44-hour cumulative pipeline
budget requests a checkpoint pause without substituting fewer examples or
partial epochs. Complete models retain selected weights, classifiers, curves
and validation results; temporary optimizer files retire after final scoring.

The current cached-CPU controls are complete. The full study recomputes matched
base embeddings and frozen controls on its numerical runtime before comparing
adapted models. Public reports expose aggregates and classifier parameters;
audio, manifests and large learned encoders remain local or in the user's
separate private Drive project. The GPU notebook reconstructs inputs from the
original archives and public field sources without needing an old feature cache.

Labels remain inherited publisher labels, without independent crop-audibility
review. Positive coverage is four, five or seven groups, not hundreds of field
trials. The 7,931 FSD recordings are reused across folds. No false alarms/hour,
event recall or detection-delay estimate is derived from these crop rates.
Candidate-source metadata and historical project exposure are recorded separately.
Exclusion from this BEATs manifest does not establish a project-wide fresh test.
