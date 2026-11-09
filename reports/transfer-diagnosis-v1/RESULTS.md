# What the repaired CNNs use and what remains unexplained

The evidence supports a narrower diagnosis than a general inability of scratch
CNNs to learn spectrograms. The existing repaired networks can learn the
simplified within-session published-label task well. Their source transfer and
their use of the desired physical distinctions remain unresolved. The paired
normalization learning results are retained separately under
`reports/learnability-normalization-v1/`.

## Specific observations

Four already-assessed recordings were selected by a fixed disagreement/confounder
rule. Both the first and middle full three-second context were used, with the
same waveform for each model. All labels retain their publisher meaning; none
is upgraded to a listening-verified audible interval.

| Case | Observation | What it establishes |
| --- | --- | --- |
| FPV `fpv_53_1`, first context | Seed42 scores0.406 and seed7 scores0.064. Time-mean replacement and either time permutation keep the scores near0.411 and0.065. | The missed classification in this context does not require coordinated temporal structure to persist. The frequency/time tensor is available; these responses are largely insensitive to the tested time changes. |
| ITU interval00, first context | Seed7 scores0.934, versus0.171 for seed42. Seed7 scores0.964 after per-band time-mean replacement and about0.961 after independent band-time shuffling. | This high response does not require the original temporal variation or inter-band synchronization in this particular input. It does not prove all learned cues are static or that a specific acoustic cue is causal. |
| Same ITU interval, middle context | A clear multi-line moving structure is visible in the high-resolution STFT and remains visible in the actual 256-band model input. Scores fall to0.076 and0.036. | Visible fine structure survives preprocessing but does not guarantee a high score. The publisher-positive recording alone does not establish the audible truth of either context. |
| FPV `grass_trimmer_1`, middle context | Both CNNs give low scores,0.011 and0.003. | A selected mechanical negative is correctly rejected; this one case does not establish general rotor/machinery discrimination. |

The figures retain the actual waveform, a genuinely 4096-sample STFT at a common
numerical power reference, the actual model tensor, genuine Grad-CAM and all
probe scores. Uniformly scaling a context by0.25 leaves the peak-relative CNN
scores effectively unchanged, as expected. Scaling only its second half changes
the within-context distribution: seed7's first ITU score falls from0.934 to0.509.
Uniform gain and a within-context level transition are different interventions.

The tensor permutations are deliberately nonphysical diagnostics, not new audio
accuracy trials. A common permutation changes local order while retaining
cross-band simultaneity; independent per-band permutations additionally remove
that synchrony. Both preserve each frequency row's marginal distribution. The
non-neural SVM's temporal mean/std representation is invariant to these
permutations; its score invariance was independently checked to1e-5 on the cases.
Such invariance is a model property, not evidence of acoustic usefulness.

## A zero Grad-CAM map is not evidence of no input dependence

Both first-ITU drone-logit positive Grad-CAM maps are zero, including the model
scoring 0.934. A disclosed follow-up occluded every consecutive32-row frequency
region in16-row steps on all eight contexts. Removing roughly 79–990Hz reduces
that seed7 score to0.00084. This demonstrates numerical dependence on the low
input region despite the zero positive Grad-CAM. Spatial gradient averaging and
positive-only clipping are not a complete explanation of the classifier.

Occlusion sets the region to the existing numerical input floor without
renormalizing the rest. It creates unfamiliar, nonphysical inputs. The result
does not prove that the removed content was audible drone sound or that the
network uses a correct physical criterion. Every position, case and both seeds
are retained, rather than only the strongest removal.

## Thresholds explain only part of the transfer failure

On the consumed FPV set, the two-hit activation AUC is0.839 for seed42,0.732 for
seed7 and0.676 for the SVM. At a threshold above every FPV negative activation,
seed42 can retain8/21 positives, seed7 only2/21, and the SVM none. YAMNet has a
negative recording with saturated activation1, so no admissible threshold in
(0,1] achieves zero negative-file alarms. This is a within-set oracle, not a
calibration promise or independent result.

A single shared threshold respecting the provisional5% negative-file budget
separately on original development and consumed FPV gives:

| Fixed CNN weights | Shared on-threshold | Earlier development TP /22; FP /188 | Consumed FPV TP /21; FP /16 |
| --- | ---: | --- | --- |
| Seed42 | 0.882244 | 16;8 | 8;0 |
| Seed7 | 0.865878 | 17;9 | 0;0 |

Both retain two-hit confirmation, two off-hits and the0.75 off/on ratio. The
seed42 adjustment removes the FPV false positive while retaining its eight
detections and earlier development recall. It leaves13/21 FPV positives missed.
The earlier development constraint prevents lowering seed7 enough to help FPV.
This is a bounded consumed-data decision control; no new threshold is promoted.
It cannot account for the whole CNN source gap.

## Normalization distinctions

- Context-peak normalization divides each power tensor by its own maximum.
  A common waveform gain therefore cancels. This was already established in
  earlier audits and is confirmed on the new cases.
- A fixed STFT power reference retains numerical recording level. It does not
  provide physical sound-pressure calibration across microphones.
- RMS normalization scales each available input toward a shared numerical RMS.
  An exact version also cancels a uniform waveform gain. It changes the reference
  from a maximum spectral bin to waveform energy; it is not background removal
  or an SNR improvement.
- The old20dB/peak caps let only 3,194/5,491 clips reach RMS 0.05. Raising the gain
  ceiling to40dB still misses 70 because of peak caps. The new exact numerical
  control retains the silence guard but does not cap or clip non-silent float
  amplitudes. Its metadata declares `normalize_rms: "exact"`; the existing
  boolean policies preserve their prior behavior.

The earlier fixed-reference learning runs used the old objective, so they cannot
settle this question for the repaired training recipe. The new controlled task
keeps the 154,689-parameter architecture and compares amplitude rules using
matched initial weights, batch orders, released labels and two seeds.

## Limits and next discriminating evidence

The simplified task changes the target to the publisher-released one-second
class labels and removes much of the source variation. Its success cannot
attribute the original problem solely to labels, normalization, architecture,
data volume or domain shift. Scalar controls and a credible train-fitted
non-neural comparison are included to avoid equating high CNN accuracy with
learning harmonic relations.

The strongest remaining question is whether the learned distinctions remain
useful across capture conditions when audible presence and absence are actually
reviewed. Richer views and3D inputs stay deferred. NEW-DATA.md records why the
inspected human-annotated releases still do not supply a complete fresh
audible-event benchmark. No listening judgments, unseen negative labels or
field-reliability claims were invented to close that gap.
