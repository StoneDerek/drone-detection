# First completed adapted arm: assessment

**This arm does not show a clear reliability gain.** At the prescribed DEV
calibration rule, fine-tuning detects three additional AuDroK positive crops
while increasing false positives on the FSD background challenge.

| Metric | Matched frozen control | Adapted encoder + refitted C1 classifier |
|---|---:|---:|
| AuDroK weak-positive recall | 31/58 — 53.45% | 34/58 — 58.62% |
| FSD negative-clip false-positive rate | 97/7,931 — 1.22% | 141/7,931 — 1.78% |

All 31 previously detected AuDroK clips remain detected, three misses are
corrected, and 24 remain missed. On FSD, adaptation corrects 40 previous false
positives but introduces 84 new ones; 57 are shared errors. These paired counts
describe changing error identities, not a new independence or significance claim.

The gain in positive recall is **5.17 percentage points**; the increase in FSD
false-positive rate is **0.55 percentage points**. Different deployment costs and
prevalence could value that tradeoff differently. This result alone does not
justify promoting the adapted model as a more reliable detector.

## What the classifier refit resolved

The selected epoch-3 joint-head DEV loss was 0.823175. Refitting the TRAIN scaler
and the same C1 logistic classifier reduced DEV loss to **0.588679**, close to
the frozen control's **0.583962**. Therefore, the earlier joint-head deterioration
overstated the difference in the final classifier's DEV loss. The final held-source
comparison still shows the recall/FPR tradeoff above.

All three epochs and **7,608 optimizer steps** completed. The selected encoder
has actual weight updates; the final head was fitted only on its TRAIN embeddings.
Each model retains its own threshold from the same predeclared DEV rule. No
held-source threshold, learning rate or data mixture was tuned for this result.
Cached selected-encoder features and the saved joint head reproduce the recorded
DEV loss within **3.18 × 10⁻⁸**. Reconstructed confusion counts and cutoff values
match the completed evaluation receipts exactly.

## Limits and next evidence

The validation pool contains all **7,989 prescribed clips**, but the 58 AuDroK
positives belong to one conservative campaign group. The 7,931 FSD clips belong
to 2,122 uploader groups and are reused project material. Publisher labels are
weak; central-crop audibility has not been independently reviewed. These results
do not establish event recall, false alarms/hour, near-human performance or broad
field generalization.

One of four adapted arms is complete. The next arm adds WSU positives to the
AuDroK-held training mixture under the same predeclared protocol. Its matched
frozen control is already available. Completing that comparison will help assess
whether positive coverage changes this result. It will not turn the reused
collections into a fresh confirmation set. The cumulative 44-hour budget remains
in force.

See [the complete paired report](README.md), [numeric evidence](paired-comparison.json)
and [runtime-control provenance](../../../MATCHED-RUNTIME.md). Classifier parameters
and training-completion receipts are retained under `heads-fine-tuned`; the large
selected encoder and complete per-record paired ledger remain in private study
storage. No new FPV or DADS audio has entered this experiment.
