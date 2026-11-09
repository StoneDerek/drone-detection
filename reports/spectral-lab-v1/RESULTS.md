# Spectrogram and non-neural development experiments

All results below use the same 405 real training files and 210 development files. The 22 positive development files cover four provenance groups. These are development results; no new final assessment or replacement of the assessed default is claimed.

The [versioned manifest and splits](../real-v2/manifest-snapshot.jsonl) and [primary source versions](../real-v2/SOURCES.md) are shared with the preserved v2 protocol.

## What changed

Six CNNs share the same 1,781-parameter architecture: 6-filter 5×5 convolution, 8-filter 3×3 convolution, 16-filter 3×3 convolution, ReLU, global average pooling, dropout 0.1 and a scalar linear logit. Frequency/time strides are (2,2), (4,2), (4,2). All ran eight epochs with seed 42, Adam 0.001, L2 1e-4, clipped gradients and the same balanced recording-bag objective. Positive bags use their maximum sampled logit; negative bags penalize all sampled windows. The best balanced development bag-loss checkpoint was retained. Only one seed was explored.

At most 60 uniform contexts per training file supply 4,648 candidate inputs, including 1,262 contexts from positive bags and 3,386 from negatives. All were visited by epoch 8; an earlier selected checkpoint saw fewer. Per-model history records that exact count. Publisher-positive windows are not all assigned positive labels.

| Arm | Actual frequency × time × channel tensor | Hann window | Intensity |
| --- | --- | ---: | --- |
| linear-fft1024 | [483, 184, 1] | 64 ms | log_power |
| linear-fft4096 | [1933, 172, 1] | 256 ms | log_power |
| linear-power-fft4096 | [1933, 172, 1] | 256 ms | linear_power |
| mel256-fft4096 | [256, 172, 1] | 256 ms | log_power |
| mel64-fft1024 | [64, 184, 1] | 64 ms | log_power |
| mel64-fft4096 | [64, 172, 1] | 256 ms | log_power |
| linear256-fft4096 (matched-axis follow-up) | [256, 172, 1] | 256 ms | log_power |

FFT spacing is 15.625 Hz for 1,024 samples and 3.90625 Hz for 4,096 samples at 16 kHz. The actual Hann window is lengthened; this is not zero-padding or image enlargement. The 256-sample STFT hop is 16 ms. Every model sees 50–7,600 Hz in the same 3-second context with a 0.5-second detection step. The longer analysis window trades temporal discrimination for frequency discrimination. Mel filtering averages bins; logarithmic intensity is a separate operation. All inputs use per-context relative power, with an 80 dB floor for log inputs. Absolute capture level and frequencies above 8 kHz are not studied here.

The non-neural RBF SVM uses the high-resolution linear-frequency/log-power input, then temporal mean/std at each of 1,933 frequency bins: 3,866 descriptors. Weighted standardization is fitted on training only. Four multiple-instance updates select positive contexts and retain all sampled negatives. C=1, gamma=scale. Its support vectors, dual coefficients, intercept and normalization are numeric NPZ arrays, and its portable scores were checked against sklearn.

One subsequent architecture probe adds a frequency-only 16-filter 5×1 convolution to the high-resolution linear/log-power CNN: 3,077 parameters. The frequency receptive field expands from 25 bins (about 98 Hz) to 153 bins (about 598 Hz), while temporal receptive field stays fixed. This addresses the fact that denser inputs shrink the physical span of fixed pixel kernels. Its separate follow-up plan, training history and source snapshot are retained; it was motivated after the original sweep, not part of its initial planned comparison.

A separate matched-axis control uses 256 area-normalized triangular bands uniformly spaced in Hz. It matches the mel256 tensor, initialization, CNN capacity, FFT, amplitude rule and training recipe. This avoids attributing the full 1,933-bin FFT advantage solely to frequency warping when it also retained more bins. Its own follow-up plan and source snapshot distinguish this diagnostic extension from the original sweep.

Package validation caught a trained status inherited by the two calibrated follow-ups. The corrected packages explicitly mark frozen decisions; numerical weights, decisions and results did not change. The [correction record](freeze-status-correction.json) retains old/new identities and equality checks.

## Coarse grid and additional calibration

The initial predefined threshold grid produced zero detections for these small CNNs at the 5% negative-file budget, despite score ranking above chance. Those frozen packages and results are retained under `coarse-grid/`. A separate, explicitly additional development calibration used negative-recording activation statistics to set a threshold for one- and two-hit confirmation, with a 1e-5 numerical margin. Neural weights did not change. Its development FPR is constrained by construction, and does not establish future FPR.

| Model | Coarse-grid TP / 22 | Coarse-grid FP / 188 | Calibrated TP / 22 | Calibrated FP / 188 | Max-score ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| frequency-context-fft4096 | n/a | n/a | 11 | 9 | 0.863 |
| linear-fft1024 | 0 | 0 | 11 | 9 | 0.830 |
| linear-fft4096 | 0 | 0 | 11 | 9 | 0.867 |
| linear-power-fft4096 | 0 | 0 | 2 | 9 | 0.774 |
| linear256-fft4096 | n/a | n/a | 12 | 9 | 0.835 |
| mel256-fft4096 | 0 | 0 | 10 | 9 | 0.848 |
| mel64-fft1024 | 0 | 0 | 2 | 9 | 0.669 |
| mel64-fft4096 | 0 | 1 | 2 | 9 | 0.721 |
| svm-linear-fft4096 | 13 | 2 | 15 | 9 | 0.849 |

Existing controls on the same development recordings: frozen YAMNet + logistic regression **20/22 positive files detected, 9/188 negative files alarmed**; non-neural relative-mel summaries + logistic regression **18/22, 9/188**. Those controls have separate previously frozen thresholds. These comparisons do not isolate neural networks as the sole cause: representations, pretraining, capacity and optimization differ.

Linear high-resolution log-power improved ranking over the coarse 64-mel CNN in this one-seed pilot. Linear power had worse ranking than log power at the same high-resolution frequency grid. This supports preserving frequency detail for further study while retaining log intensity. It does not establish that every high-resolution tensor or non-neural method is better. At matched 256×172 dimensions, uniform-Hz bands detect 12/22 versus 10/22 for mel, both at 9/188 development false positives; their ROC AUCs are 0.835 and 0.848. The full 1,933-bin linear FFT detects 11/22 with higher ROC AUC 0.867. More bins therefore did not monotonically improve recall at this operating point. The frequency-context extension still detects 11/22. The small CNN budget, weak labels, few positive groups and changed physical receptive fields limit the conclusion. No expensive architecture expansion or default promotion follows.

## Source results and failures

### linear256-fft4096

| Source | Positive files detected | Negative files alarmed |
| --- | ---: | ---: |
| batear-field | 1/11 | 0/6 |
| d3set-ground | 11/11 | 0/0 |
| d3set-noise | 0/0 | 4/22 |
| esc10 | 0/0 | 5/160 |

Group-bootstrap interval: `{'false_positive_fraction': [0.011234385788713828, 0.09183673469387756], 'recall': [0.09090909090909091, 1.0], 'warning': 'Conservative source/day groups, not independent windows. Few positive-source groups make intervals inadequate for field claims.'}`

### svm-linear-fft4096

| Source | Positive files detected | Negative files alarmed |
| --- | ---: | ---: |
| batear-field | 4/11 | 4/6 |
| d3set-ground | 11/11 | 0/0 |
| d3set-noise | 0/0 | 2/22 |
| esc10 | 0/0 | 3/160 |

Group-bootstrap interval: `{'false_positive_fraction': [0.016042780748663103, 0.08629921799972558], 'recall': [0.36363636363636365, 1.0], 'warning': 'Conservative source/day groups, not independent windows. Few positive-source groups make intervals inadequate for field claims.'}`

## Interface and reproducibility

Run `bash scripts/start_lab.sh`. The local lab accepts audio, compares selected models, plays/seeks score timelines, and shows the selected CNN input beside genuine Grad-CAM. Choose the logit class, convolution layer, context and frequency zoom. Training histories, selected epochs, actual tensor dimensions and development errors are visible. Full comparison and explanation data can be downloaded as JSON. Grad-CAM is available for the trained spectrogram CNNs; it is not fabricated for the SVM or opaque YAMNet SavedModel.

Attribution uses gradients of the selected CNN logit, not the input amplitude as a proxy. Its native spatial grid is disclosed; interpolating that grid adds no explanatory detail. A heatmap alone does not prove a causal or useful acoustic cue. Zero maps remain zero. The browser demonstration uses the fixed existing Batear development recording `batear-20260415_192501` (9.078 seconds); it supplies software evidence, not an additional trial.

Complete coarse and calibrated models, source snapshots, plans, data identity, per-source outcomes, every compressed development score trace and batch/sequential parity checks are retained. Raw audio and multi-gigabyte feature caches remain in ignored `runs/`. The initial slow dense mel projection was replaced before remaining arms with an equivalent sparse multiplication; 16 checked contexts matched exactly. Completed arms were preserved.

Archived raw score traces retain their parent model identity and original/placeholder alarm state. The calibrated decisions and resulting recording alarm intervals are in `calibrated-results.json`; apply those decisions to the raw scores to reproduce outcomes.

The original v2 assessment remains 100/133 publisher-positive files detected and 126/1,608 negative files alarmed for the unchanged YAMNet default. All later spectral experiments are development work. Event recall, true false alarms/hour, acoustic range, physical capture delay and live microphone operation remain unvalidated.
