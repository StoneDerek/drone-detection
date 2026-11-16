# Frozen BEATs classification heads

These are trained linear classification heads, not complete encoder checkpoints.
The pretrained BEATs encoder remains frozen. Each model.json binds the exact
encoder checkpoint, code, native frontend, package versions and head SHA256.
The three variants use the same 768-dimensional encoder representation.

Load head.npz with `numpy.load(..., allow_pickle=False)`. For embedding `x`, the
raw score is `((x - mean) / scale) @ coef + intercept[0]`. Predict drone when the
score is at least `model.json`'s `cutoff_logit`. These weighted-training logits
are not calibrated real-world drone probabilities.

Original and recalibrated contain byte-identical heads with different cutoffs.
Confuser-trained contains the new head trained on 16,048 crops. All three are
research artifacts: none has demonstrated reliable field performance. See the
[findings](../../reports/confuser-study-v1/FINDINGS.md) and
[complete source-specific results](../../reports/confuser-study-v1/RESULTS.md).
