# BEATs score ranking on the two development folds

Fine-tuning helped the first AuDroK comparison near its original operating point.
At the same cap of **97 FSD false positives**, the adapted model detects **33 of
58** AuDroK drone clips, up from **31 of 58** for the frozen model. At the
strictest setting that advantage reverses. With zero FSD false positives in this sample,
the frozen model detects **17** clips and the adapted model detects **5**.
Across all cutoffs, AuDroK versus FSD ROC AUC falls from **0.836 to 0.720**.

The original evaluation applies cutoffs selected on DEV. It found 31 detections
and 97 FSD false positives for the frozen model, and 34 detections and 141 FSD
false positives after fine-tuning. The score ranking comparison below uses
observed validation scores to compare at equal false-positive counts. This is
an exploratory comparison. A future evaluation still needs a cutoff selected
before inspecting its test scores. [Original paired assessment](../paired-analysis/audrok-dresden/current/INTERPRETATION.md)

| FSD false positives out of 7,931 | AuDroK detections with current frozen | AuDroK detections with current fine-tuned | AuDroK detections with broader frozen |
|---:|---:|---:|---:|
| 0 | 17 | 5 | 0 |
| 39 (0.5% cap) | 29 | 30 | 1 |
| 79 (1% cap) | 30 | 33 | 12 |
| 97 (current frozen model's original count) | 31 | 33 | 14 |
| 141 (adapted model's original count) | 31 | 34 | 18 |
| 158 (2% cap) | 31 | 34 | 18 |
| 396 (5% cap) | 36 | 35 | 25 |

The first adapted model improves ranking at several budgets near 1%. It ranks
fewer positives above the highest scoring background clip, and its lower AUC
shows poorer ranking overall. The broader frozen model also illustrates why full
AUC can mislead here. Its AUC is slightly higher than the current frozen model's
(0.840 versus 0.836), while recall is lower at every shown false-positive cap.
The broader adapted model is still training.

## Local background produces a different ranking result

The WSU fold has 1,486 drone crops and 190 ambient-tail crops. Both frozen models
achieve about **0.991 AUC** when WSU drone crops are compared with FSD negatives.
On WSU positives versus WSU background, AUC is **0.836** for the current mix and
**0.853** for the broader mix. With both models set to a cap of 79 FSD false
positives, the current mix detects 1,268 drone crops and fires on 88 WSU background
crops. The broader mix detects 1,281 and fires on 83 background crops.

At that matched point the broader mix helps slightly on WSU and hurts on AuDroK.
The WSU background errors show why FSD and local ambient audio need their own
results. The WSU ambient crops come from one recording, so this gap describes
that recording. The AuDroK positives belong to one conservative campaign group.
Publisher labels are weak and the clips share capture conditions. These counts
describe clip decisions on material already used in project development.

## Reproduce

Run from the repository root with the prepared CPU study and locked environment:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/diagnose_beats_score_ranking.py \
  --work work/beats-adaptation-cpu2-v1 \
  --output work/beats-score-ranking-v1/diagnostics.json
```

The script verifies the cohort, fitted heads, feature matrices and completed
receipts. It reconstructs the original confusion counts exactly before comparing
rankings. Each cap excludes any score tie that would exceed the permitted FSD
count. The archived run covers all four frozen controls and the one completed
adapted result. It uses 7,989 and 9,607 validation rows for AuDroK and WSU,
respectively. Later runs will include additional adapted arms as they finish.
The diagnostic uses cached embeddings throughout. The output records counts,
AUC and provenance hashes. [Numeric diagnostics](diagnostics.json)
