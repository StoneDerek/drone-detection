# Model lab validation

Product 0.3.0 supplies an interactive comparison and attribution lab. All new
CNN/SVM results are development evidence. The original assessed YAMNet default
is preserved and remains stronger than these new small CNNs.

- **66 software tests passed**, including spectral axes/intensity, actual tensor
  dimensions, both CNN architectures, weak recording-bag training, numeric model
  loading, artifact integrity, calibration, trained-to-frozen transitions,
  file/stream and batch-context agreement, genuine gradient attribution, upload
  comparison, input handling and unavailable/inapplicable explanations.
- Static and repository whitespace checks passed. The wheel built successfully;
  every shipped Python module matched both the working source and its isolated
  installed copy byte for byte.
- The complete catalog loaded **11 models**: eight new spectrogram CNNs, the
  new non-neural RBF SVM, the existing YAMNet default and the existing non-neural
  relative-spectral/logistic baseline. Eight models expose genuine Grad-CAM.
- Chromium passed real upload comparison, playback/seek, model scores/alarms,
  classifier-score agreement with attribution, class/layer/context controls,
  frequency zoom, visible training curves, model identities, JSON exports and
  actionable invalid-audio handling. Changing targets clears stale attribution.
  No browser page errors were observed.
- A new virtual environment installed hash-checked runtime dependencies and the
  0.3.0 wheel. Python `-I` launched from `/tmp` with copied CNN/SVM packages,
  catalog and one input file. The installed module came from that environment's
  `site-packages`. No editable source, evaluation corpus or feature cache was
  needed. That isolated server passed the same browser workflow for both class
  attribution targets.
- The unchanged default was checked against the prior 0.1.0 isolated-wheel trace
  on the same existing real development input. All 14 windows matched exactly
  in score (maximum error 0), alarms, timestamps, padding and input reasons.
- Original and additional calibration results are both retained. The calibrated
  models' numeric arrays were checked equal to their respective parent arrays.
  Batch/sequential score checks have a 1e-5 tolerance; model-specific errors and
  the additional threshold margin are recorded with the calibration.
- Strict loading rejected the two follow-up drafts whose calibrated metadata
  still inherited a `trained` flag. The explicit freeze correction retained the
  rejected drafts, changed no numeric weights/decisions/metrics, and records old
  and corrected identities. The regression tests now load frozen calibrated
  artifacts from trained parents for both architectures.

The fixed browser input is `batear-20260415_192501`, 9.078 seconds, publisher label
drone, development split. It was already used for software verification before
this model sweep; it was not selected because the new CNN succeeded. Its SHA-256
is `18ef2eb6318f075bd1c978b2c0c16dcb01fd8a6575925703e0fca27f0206395c`.

The selected small CNN produced **No alarm** on this publisher-positive example
while the YAMNet control alarmed. Its selected drone-logit attribution was zero;
the interface showed that zero map. Switching to the non-drone logit produced
positive attribution. Both outcomes are retained. No input-intensity proxy or
invented heatmap was substituted for a zero gradient attribution result. This is
an observed model/label disagreement, not independent confirmation of audibility
or a causal explanation of the error.

[Full catalog browser check](browser-verification.json) ·
[Isolated install](clean-install-verification.json) ·
[Isolated browser check](clean-browser-verification.json) ·
[Default regression](default-regression.json) ·
[Original zero-map screenshot](product-screenshot.png) ·
[Installed lab, non-drone target](clean-product-screenshot-non-drone.png)

The software checks use generated signals where appropriate; these are not
acoustic reliability trials. Field evidence remains limited by publisher labels,
few positive provenance groups, unmatched capture conditions and missing
independently reviewed event boundaries. The new default-sized neural inputs,
larger tensors and Grad-CAM interface do not establish a range, SNR tolerance,
wind limit, event recall, true false alarms/hour or live-microphone capability.
No environment publication or external deployment was performed.
