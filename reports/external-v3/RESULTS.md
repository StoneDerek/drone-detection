# Frozen external-source assessment v3

Four pre-existing models were evaluated at their existing frozen decisions. No weights, amplitude rules or thresholds were tuned on these recordings. Results vary strongly by source; this assessment does not establish field reliability.

| Frozen model | FPV positives detected /21 | FPV negative files alarmed /16 | ITU eligible positive intervals detected /34 |
| --- | ---: | ---: | ---: |
| coherent-frequency-seed42 | 8 | 1 | 4 |
| coherent-frequency-seed7 | 0 | 0 | 20 |
| svm-linear-fft4096 | 1 | 4 | 31 |
| yamnet-default | 20 | 8 | 12 |

## What those counts measure

FPV contains 37 publisher-labelled recordings (21 positive, 16 negative), totaling 1.414 h. Its session ancestry is undocumented and conservatively forms one source component. These are not 37 independently verified capture sessions. Source attribution and limitations are in SOURCES.md.

ITU contains 35 manually selected positive intervals from one original session, plus 1,000 one-second background clips. Only 34 positives provide a full 3 s context plus a second confirmation score. The remaining 1 s positive and all background files are retained in raw replay results but excluded from the comparable full-context cohort.

Every model gives zero alarms on those one-second backgrounds because the frozen decision needs two hits. That is a buffering/confirmation consequence, not evidence of zero field false positives. There is no eligible continuous negative denominator for ITU. The raw max-score AUCs also mix long positive intervals with short padded negatives, so they are not treated as a fair matched-duration discriminator benchmark.

The two sources were absent from the task-specific training, repeatedly reused development and consumed v2 assessment. No exact/raw, canonical-decoded or gain-normalized whole-file matches to the 2,356 earlier recordings were found. That cannot prove the absence of unknown shifted excerpts, lossy copies or all audio-pretraining overlap. Each source is one conservative provenance component, so file/window bootstrap intervals would overstate independent evidence.

Labels are publisher evidence, not newly invented listening judgments. ITU reports two annotators and kappa 0.5869; its background comes from unlabelled stretches. Reviewed audible event boundaries, truly negative continuous stretches and independent new capture sessions remain missing. Event recall, true false alarms/hour, physical capture delay and range remain unassessed.

## Concrete alarm failures

- coherent-frequency-seed42: itu-aris: {'all_scores_below_on_threshold': 29, 'peak_crossed_but_confirmation_not_met': 1}; saarmets-fpv: {'all_scores_below_on_threshold': 13}
- coherent-frequency-seed7: itu-aris: {'all_scores_below_on_threshold': 10, 'peak_crossed_but_confirmation_not_met': 4}; saarmets-fpv: {'all_scores_below_on_threshold': 20, 'peak_crossed_but_confirmation_not_met': 1}
- svm-linear-fft4096: itu-aris: {'all_scores_below_on_threshold': 3}; saarmets-fpv: {'all_scores_below_on_threshold': 19, 'peak_crossed_but_confirmation_not_met': 1}
- yamnet-default: itu-aris: {'all_scores_below_on_threshold': 18, 'peak_crossed_but_confirmation_not_met': 4}; saarmets-fpv: {'all_scores_below_on_threshold': 1}

These mechanisms describe the frozen alarm engine: either all scores remain below its on-threshold, or isolated crossings fail confirmation. They do not diagnose whether an acoustic cue was audible or why a model assigned that score. review-cases.json retains the first two false positives/negatives by sorted original ID per source/model, plus one correct positive/negative where present. Each case includes its exact audio hash, peak context, decision and alarm intervals. Those examples were not selected to make a model look favorable.

## Separate small published-label task

The predeclared one-second non-neural task is documented in ../human-label-task-v1/RESULTS.md. It matches class durations, uses only complete STFT frames, train-only statistics and publisher interval-disjoint assignments. At fixed 0.5 it detects 10/10 test positive intervals and 0/3 background blocks, but only 183/674 positive-labelled seconds (AUC 0.888). This shows the importance of the target and aggregation: a single high-scoring second can satisfy an interval label. It is same-session feasibility, not a replacement for fresh-session validation or a direct model-versus-model comparison.

## Reproducibility and next evidence

panel-plan.json fixes model IDs, preprocessing and decisions before scoring. compatibility-amendment.json records the input-duration issue and the matched-duration task before external scoring. assessment/frozen-execution.json fixes the manifest, source audit, runner, hardware and plan hashes. assessment-use.json consumes this cohort before inference. The recovered CNN packages and 32 prior-fixture comparisons are described in RECOVERY.md.

All compressed timestamped scores, per-recording outcomes and models are retained. Run scripts/summarize_external_v3.py to reproduce every saved recording outcome from the frozen decisions without scoring audio again. The one-use evaluation runner refuses a second assessment in place. Its numerical results are now consumed for any subsequent model optimization.

The next discriminating evidence is another independently captured session with matched drone/background conditions, manually reviewed audible intervals and explicit ambiguous spans. The source-dependent reversals do not support declaring one architecture family inherently best. The fixed one-second task shows that this spectral input can carry useful published-label distinctions, while leaving cross-session generalization unresolved.
