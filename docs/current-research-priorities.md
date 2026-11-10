# User priorities and current research state

The central objective is to understand what distinguishes drone audio, verify
that the pipeline preserves it, and establish that the model learns it before
adding complexity. A working detector and an explanation of its failures remain
the goal. This note records the user's recovered preferences and links them to
the current evidence; it is not a claim that the research problem is solved.

The user's latest direction supersedes another small matched-case or
same-session study as the next milestone: the current results are unacceptable,
and the next phase must secure substantially larger, independently varied real
data and use literature-based baselines. Follow
[the data-first plan](research-plan-v2.md), including source/label/coverage checks
and a reserved continuous benchmark. The previous 100-hour and diversity-count
budgets were unvalidated proposals and are withdrawn as readiness requirements.
Training quantity will be assessed with grouped learning curves; evaluation
exposure is sized separately for uncertainty and coverage. The
[broader source audit](../reports/source-scope-audit-v3/README.md) covers all ten
user-identified original sources. The small audited AuDroK subset does not
establish the amount of suitable public data available.

| Priority | Established work | Still open |
| --- | --- | --- |
| Fundamental scratch-CNN problem | Actual parameter counts, frequency-path coverage, dropout/max-loss defect, sampled-bag target mismatch, gradients, optimizer, caches, checkpoints and replay audited. Tiny CNN memorization on real examples works. | No single sufficient cause explains cross-source failures. Repairs do not prove that CNNs inherently underperform YAMNet. |
| Varied real data and stronger tests | Original405/210 train/development files, frozen FPV/ITU source check, human-primary Kenya positives, and one-channel controlled-capture recordings with explicit field backgrounds. Source pins, first-use ledgers and conservative grouping retained. | Capture dates, site/device and class remain entangled; positive recording labels do not establish audible timing. Repeated crops and extra channels are not independent evidence. |
| Fine acoustic detail | Genuine4096-sample analysis windows, linear/mel/filterbank comparisons, real tensor dimensions and physical receptive fields. | Finer input does not prove useful harmonic or motion relationships are learned. |
| Richer representations | Multi-view/context and3D CNN suggestions retained as future hypotheses. | No implementation requested; first identify useful distinctions from actual cases. |
| Normalization and quieter sounds | Peak-relative cancellation, fixed numerical reference, and capped versus exact RMS controls. Same-session learning works across rules; frozen transfer checks expose high FPV background scores under RMS. | Better positive coverage does not establish better discrimination; no physical SPL/SNR calibration follows. |
| Specific examples and simpler tasks | Real spectrograms, score disagreements, synchronized-versus-independent time probes, fixed within-session label task. | Published labels do not establish every second's audibility. Keep observations distinct from causal explanations. |
| Inspectable behavior | Cloudflare lab, retained recordings, playback, zoom, selectable Grad-CAM and exports are established. | A heatmap is not proof of recognition; preserve zero maps and actual native resolution. |
| Neural versus non-neural | Frozen SVM/YAMNet/CNN source comparisons and matched-input weighted logistic controls. | Any observed advantage is conditional on data, target, representation and operating point. |

The [cross-condition study](../reports/cross-condition-v1/RESULTS.md) follows the
within-session control with fixed models, source-separated decisions and exact
non-neural frequency contributions. It does not promote a detector. The next
training comparison needs both classes within varied capture conditions and a
group/source holdout established before fitting, alongside a concrete check of
the desired harmonic distinctions against broad-spectrum alternatives.

Published human annotations are preferred over asking the user to label a batch.
Do not reset old project files, republish the environment, invent listening
judgments, or implement deferred richer-input ideas as an inferred request.
