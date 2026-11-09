# Scratch-CNN learnability and amplitude-reference controls

The repaired scratch CNN learns this simpler published-label task well at both seeds. A fixed numerical amplitude reference or exact common RMS does not establish a general cure for source transfer. The input duration, target and recording conditions differ from the original weak-recording detector; comparisons are descriptive on an already-consumed session.

| Input rule | Model | Seed | Selected epoch | Test TP /674 | Test FP /150 | Segment ROC AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| peak | CNN | 42 | 8 | 673 | 2 | 0.9996 |
| peak | CNN | 7 | 22 | 673 | 2 | 0.9997 |
| peak | 512descriptor logistic | 42 | — | 665 | 2 | 0.9987 |
| fixed | CNN | 42 | 8 | 672 | 2 | 0.9997 |
| fixed | CNN | 7 | 24 | 666 | 1 | 0.9936 |
| fixed | 512descriptor logistic | 42 | — | 663 | 2 | 0.9924 |
| rms-fixed | CNN | 42 | 24 | 672 | 1 | 0.9994 |
| rms-fixed | CNN | 7 | 20 | 672 | 2 | 0.9994 |
| rms-fixed | 512descriptor logistic | 42 | — | 657 | 2 | 0.9964 |
| exact-rms-fixed | CNN | 42 | 12 | 672 | 1 | 0.9994 |
| exact-rms-fixed | CNN | 7 | 20 | 671 | 2 | 0.9994 |
| exact-rms-fixed | 512descriptor logistic | 42 | — | 655 | 2 | 0.9964 |

## What was held fixed

All inputs are the same original one-second publisher segments:16kHz mono,4096-sample Hann STFT,256-sample hop,256 uniform-Hz bands over50–7600Hz,47 complete frames and no zero padding. The CNN is the existing 154,689-parameter frequency-aware architecture with dropout0. Both seeds use identical initial arrays and batch permutations across amplitude arms. Adam0.001, per-tensor clipnorm1, convolution L2=1e-4, batch64 and24epochs are fixed. Only the amplitude rule differs across paired CNN runs.

Training uses the publisher-released one-second class labels directly, with equal original interval/block mass within each class and equal class mass. This is an explicitly simpler target than maximum-instance weak recording training. The labels are not newly verified audible truth. The task has one session; publisher interval/block splits and exact half-second overlap checks are retained. Validation selects checkpoints by weighted segment BCE every2epochs. Threshold0.5 is fixed.

The non-neural controls use train-only standardized temporal mean/std at each frequency band,512 features, identical labels/weights, logistic C1 and threshold0.5. They retain less temporal information than the CNN. Any score difference is conditional on this representation and target, not an inherent neural-versus-non-neural ordering.

## What the normalization names mean

peak divides each power image by its own spectral maximum. fixed uses unit STFT power as one common numerical reference. rms-fixed first targets waveform RMS 0.05 with20dB/peak caps; only 3194/5491 inputs reach it. exact-rms-fixed removes those caps for an explicit numerical feature-control experiment while keeping the silence guard. All 5491 inputs reach the target to numerical precision. Some normalized float peaks exceed1 and are not clipped; this is a mathematical feature transform, not PCM playback or physical SPL calibration.

The exact-RMS follow-up was added after the amplitude-coverage audit, not selected from accuracy results. Its new optional normalize_rms="exact" mode is declared in each model. Existing boolean normalization policies retain their old formula and serialized default values. Original and follow-up execution code and input identities are archived.

## Interpretation limits

The source contains35 manually selected positive intervals from one session. Exported background labels come from unlabelled stretches, and annotator disagreements are not supplied. Test segments overlap and are not674 independent positive captures. Every split has already been used diagnostically; there is no fresh independent test here.

This establishes learnability of the released labels under restricted conditions. It does not isolate why the original detector fails: the simplified task changes source variation, context duration and target simultaneously. Across amplitude arms, those factors are controlled, but even a better within-session point result would not establish improved cross-source detection.

Scalar volume/brightness controls are in ../transfer-diagnosis-v1/scalar-controls.json. Raw log-RMS alone reaches test AUC0.5968 and peak-relative mean-image brightness0.6964, so those single scalars do not reproduce the high paired-model ranking. Strong descriptor-logistic results and temporal structural probes also prevent interpreting high CNN accuracy as proof of learned harmonic motion.

The script produced each arm’s test report after its validation-selected checkpoint, without using test results in any subsequent fixed arm. The execution description had said reporting would wait until all selections were complete; the actual serialized implementation and timing are retained. All recipes were fixed in advance, and the study remains explicitly consumed-data diagnosis.

No model or amplitude policy is promoted to the product. The next discriminating task needs matched recording conditions, reviewed audible on/off intervals and confounding mechanical/tonal sounds, with a separate capture condition reserved for assessment.
