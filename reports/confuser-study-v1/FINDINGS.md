# Mechanical rejection improved; field generalization remains unresolved

The full fixed comparison is complete. A new BEATs linear head trained on
16,048 crops, including 6,089 additional UrbanSound8K negatives. The encoder was
pretrained and remained frozen. Evaluation used 9,665 accepted new crops from
7,993 files: 7,931 FSD recordings, four WSU recordings, and 58 AuDroK recordings.
The positive field material spans **four date/site or campaign groups**, not
1,544 independent drone trials. All three heads and cutoffs were frozen before
fresh scores were produced.

The new head reduced FSD false positives from 899/7,931 to 73/7,931, and the
mechanical subset from 101/389 to 8/389. The change transfers beyond UrbanSound8K
parents and known linked uploaders. However, raising the original head's cutoff
reduced these counts further, to 22 and 5. That threshold-only variant lost much
more WSU recall: 864/1,486 versus 1,265 for the newly trained head and 1,433 for
the original. Additional training changes the tradeoff; it does not remove it.

The strongest evidence against calling this solved is the **WSU ambient tail**.
False positives were 163/190 for the original head, 20/190 after recalibration,
and 85/190 for the new head. Those windows come from one recording and are
publisher-labeled negatives. They do not estimate a population false-alarm
rate, but they expose a major failure in a concrete field condition. On WSU,
ranking AUC changed only from 0.8292 to 0.8357. The cutoff-only variant has exactly
the original AUC, as expected. The new training did not produce a large
improvement in separating these field classes.

AuDroK also remains difficult: original recall 33/58, recalibrated 26/58, and
new-head 31/58. Its seven publisher drone codes provide useful variation, but
all recordings belong to one campaign and central-crop audibility is unverified.
Neither the new head nor the threshold-only variant is an overall replacement
that has demonstrated reliable field performance.

These results narrow the next decision. Generic mechanical negatives and
calibration deserve a permanent place in the training/evaluation pipeline.
They do not explain or solve the remaining field failures. The experiment
does not identify shortcut learning, amplitude, microphones, or representation
quality as the single cause.

The current positive TRAIN still contains only 976 crops from two source
collections. The next milestone should separate **positive training diversity**
from **encoder adaptation**, with these controls:

1. Preserve the current models and scores. These evaluation collections are now
   consumed development evidence. Reserve additional public recording groups
   for any subsequent claim of fresh transfer; do not recycle these results as
   a new test.
2. Review the exact field crops and their surrounding public audio for audible
   drone presence, uncertainty, and confusable sounds. In particular, verify
   the WSU tail and AuDroK misses before treating publisher labels as a human
   auditory reference. A blind review inventory covering **all 1,734 selected
   field crops**, about 87 minutes of cropped audio, is included. It hides both predictions and
   publisher classes. This is label review, not a small performance benchmark.
3. Broaden the drone-positive training mix using reviewed public recordings.
   Keep campaigns, original recordings and WSU date/site groups intact in
   source-aware development folds, including their negative intervals. Keep
   within-source adaptation distinct from transfer to a held-out source.
4. Use a small controlled comparison: current versus broader positive training,
   each with a frozen encoder versus task-specific BEATs fine-tuning. The
   current frozen result supplies one baseline. Match negative training,
   frontend and operating-rule selection across comparisons; select training
   duration on development data and freeze before fresh evaluation.

| Positive TRAIN mix | Frozen BEATs + trained head | Fine-tuned BEATs |
|---|---|---|
| Current two-source positives | Existing baseline | Tests encoder adaptation with fixed data |
| Broader reviewed public positives | Tests positive diversity without backbone changes | Tests the combination |

This separates improvements from better positive coverage from improvements
that require adapting the encoder. It avoids assuming that a more trainable
model is the missing ingredient.
If verified audible positives and confusable negatives still overlap after
adaptation, a matched-resolution frontend/temporal-pooling experiment becomes
better justified.

The source and crop rules are in [PROTOCOL.md](PROTOCOL.md); numeric results are
in [RESULTS.md](RESULTS.md). The [error inventory](error-cases.csv) retains every
fresh crop misclassified by any variant. The report makes no continuous-stream
false-alarm-per-hour, detection-delay, or near-human claim.

FSD50K credit: Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic Font and
Xavier Serra, *FSD50K: an Open Dataset of Human-Labeled Sound Events*, arXiv 2020.
Per-recording Freesound creators, licenses and attribution URLs are preserved
in the result bundle. No raw audio or pretrained encoder weights are redistributed.
