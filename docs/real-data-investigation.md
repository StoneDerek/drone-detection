# What the investigation changed in the product

The historical evidence points to several possible contributors: intensity/source
shortcuts, incorrect label propagation, partial data use, inconsistent normalization
and context, and missing code/statistics. It does not isolate one definitive cause.
The implementation uses one shared preprocessor and streaming engine, train-only
statistics, explicit class mapping, source/session groups, and complete packages.
No historical model or normalization arrays were invented or substituted.

The first **new real-audio** assessment failed despite excellent development
scores. On 48 positive / 160 negative publisher-labeled recordings, the selected
relative-Mel model detected only 3 positives; absolute-Mel detected 7; YAMNet
detected 10. Those results remain saved in `runs/real-v1/results.json`. They are
not acceptable product-performance claims.

Two concrete limitations informed version 2. Training had only ten positive
flight bags from a narrow source. Development positives were long flight files,
while many unseen positives were short clips: maximizing over many windows and
then breaking threshold ties toward 0.995 favored long recordings. YAMNet's
ranking was better than spectral summaries on the consumed assessment, but the
threshold still rejected most positives. These observations support testing
broader real positives and sensitivity calibration; they do not prove a complete
causal explanation for the historical project.

Version 2 explicitly consumes the first assessment for development. The complete
Svanstrom source moves into training; all its unknown original relationships stay
together. The unknown-parent D3Set SNR background excerpt is excluded. New DDL
real-field flight excerpts supply additional training; a different collection day
is reserved. The April 28 Batear day and previously unused ESC source IDs/classes
form additional fresh assessment inputs. DDL's synthetic archive is not used.

Fragment durations come from WAV headers, not the documentation's example.
Actual inspected DDL files contain 100 ms, 96 kHz, eight-channel PCM-32 audio.
Contiguous fragments reconstruct ten-second excerpts at metadata-chosen sequence
positions; only physical microphone 0 is selected. Excerpts from the same flight
remain in the same collection-day group. They are not independent trials.

The sensitivity policy is fixed before the fresh assessment: maximize development
recording recall under a provisional 5% negative-file false-positive budget, then
prefer the lowest feasible threshold, lower false positives, and confirmation.
Candidate selection is also frozen before assessment. The original event recall,
false-alarms-per-hour and delay contract remains unfulfilled without independent
temporal/audibility labels. Publisher presence metadata and physical flight logs
are not a substitute for those labels.

The product goal is a usable file-upload, score-timeline and replay workflow with
an evaluated real-audio model. Any remaining failures and operating limits must
be visible alongside it. The portfolio page and environment publication are
outside this work.
