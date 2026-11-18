# Which FSD50K errors changed after BEATs fine-tuning?

In the first completed experiment, false positives rose from **97 to 141** at
the cutoffs chosen on development data. The existing mechanical-confuser subset
accounted for **one of the 44 net additional errors**. The remaining 43 occurred
outside that subset. The 84 newly flagged clips came from 69 uploader groups,
with at most four from any one uploader.

| FSD50K subset | Clips | Frozen false positives | Fine-tuned, original cutoff | Fine-tuned, matched count |
|---|---:|---:|---:|---:|
| All | 7,931 | 97 | 141 | 97 |
| Mechanical-confuser subset | 389 | 8 | 9 | 6 |
| Remaining clips | 7,542 | 89 | 132 | 91 |

At the matched count of 97, **47 errors are shared and 50 change in each
direction**. The largest net increases by publisher label were clothing zippers
(11 to 16 of 87) and flatulence (2 to 6 of 88). Purring (11 to 3 of 55), hissing
(6 to 1 of 105), and crackling (7 to 2 of 88) showed the largest net decreases.
The nine newly flagged zipper clips span eight uploader groups.

At the original cutoffs, clothing zippers and flatulence had 12 and 14 newly
flagged clips, respectively. Their net increases shrink from 10 and 14 to 5 and
4 when the overall false-positive count is matched. The remaining changes give
concrete cases to listen to when investigating what fine-tuning emphasized.
[Matched-count ranking comparison](../score-ranking/README.md)

The mechanical subset is the existing list of 18 publisher labels for tools,
fans, horns and road vehicles. Its complement can include other mechanical
sounds. Publisher labels apply to whole recordings, can co-occur and may describe
audio outside the scored crop. These figures describe label associations on the
development corpus. A listening review of the crops would be needed to identify
which sounds are audible. The matched count uses FSD scores as an exploratory
comparison. Original model decisions retain their development cutoffs.

All 7,931 negative clips were joined by hashed record ID to the verified paired
score ledger. The audit verifies both models' decisions against the original
receipts and emits aggregate counts for every publisher label observed in the
cohort. [Complete label table](diagnostics.csv) · [Verification hashes and totals](diagnostics.json)

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_beats_fsd_errors.py \
  --manifest work/beats-adaptation-cpu2-v1/manifest.jsonl \
  --pair reports/beats-adaptation-v1/paired-analysis/audrok-dresden/current/paired-comparison.json \
  --ledger work/beats-paired-analysis-v1/audrok-dresden/current/paired-predictions.jsonl \
  --output work/beats-fsd-error-audit-v1/diagnostics.json
```
