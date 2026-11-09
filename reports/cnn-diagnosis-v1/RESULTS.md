# CNN formulation and training diagnosis

These controls precede another architecture search. The assessed YAMNet default and all frozen comparison weights/decisions remain the same. No new final assessment was opened. See the separate [seven real case studies](../real-observations-v1/OBSERVATIONS.md) for observations that constrain the next formulation.

## Established checks

- All **7,299 sampled contexts across 615 train/development files** in the linear256 cache exactly match freshly decoded, resampled and computed tensors: maximum error **0**. Cache hashes, labels, starts, source and provenance-group identities were checked. This rules out a stale-cache/train-inference tensor mismatch for that cache, not wrong publisher annotations.
- The same **1,781-parameter CNN** can fit **16 real training contexts** (8 publisher-drone, 4 helicopter, 4 background): 100% training accuracy, BCE **0.01361**, lowest correct-class score **0.9714**, after 750 full-batch updates. This is a memorization/gradient check, not 16 independent successful detections, verified audible labels or generalization.
- Applying the YAMNet candidate's **RMS target .05, gain ceiling20dB, peak ceiling.98** before the existing per-context spectral peak normalization essentially cancels. Across one middle context per file, maximum classifier-score change was **8.94e-8**, maximum pixel difference **4.69e-5**. Only **511/615** reached the RMS target because of caps. These operations must not be described as universal equal-RMS normalization.
- With a **fixed** STFT power reference, the same gain changes the actual input: median per-file mean absolute image change **0.04065** on the 0–1 tensor scale. Reference1 means spectrum-scaled STFT power, not physical sound pressure.
- A controlled artificial image pair contains the same number and strength of narrow bands but different global frequency spacings. The original full-resolution compact CNN gives **identical scores** and pooled-feature differences below **2.4e-7**. Its receptive field is25 bins (~98Hz) before averaging. The longer-frequency variant changes features for the pair but did not solve the real source gap in the earlier experiment. This is an architecture existence proof, never a synthetic acoustic accuracy benchmark.
- A classifier using only the training-majority publisher/source category predicts **199/210** development recording labels: **11/22** positive and **188/188** negative. No audio is needed. Source categories partly encode the collection method/class; the CNN is never given these strings. This shows acquisition/class confounding that the waveform may also contain, not proof that the CNN learned a particular shortcut.
- A separate train-fitted logistic control using just **one mean-image-brightness scalar** has development AUC **.5941**. This does not reproduce the CNN's ranking. The source-only result should not be inflated into a claim that one simple amplitude/brightness cue explains its behavior.

## Controlled normalization runs

Three arms ran together with the same 405 train/210 development recordings, compact CNN, seed42, Adam.001, clipnorm1, dropout.1, L2.0001, sampled-bag objective, eight epochs and minimum balanced development bag-loss checkpoint. The baseline reused the verified original cache; the two fixed-reference arms built new caches with the same sampled starts. Plans were written before the runs. No alarm threshold was optimized or model promoted.

| Input amplitude rule | Selected epoch | Train balanced bag loss | Development balanced bag loss | Development max-sampled-context ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| Context spectral peak (repeat) | 2 | .6889 | .6765 | .8066 |
| Fixed STFT power reference, raw waveform | 8 | .6600 | .5706 | .8283 |
| Fixed STFT power reference, reference-RMS waveform | 4 | .6477 | .5591 | .8095 |

The fixed-reference variants change optimization and improve this development bag-loss measure. Adding RMS to the fixed-reference arm does not improve its ranking in this run. This does not establish a solution to the CNN failure, a universal advantage for RMS normalization, or field reliability. Labels remain weak, there are very few positive provenance groups, and only one seed was tested. The comparison uses the capped cached contexts, not full-recording event recall or false alarms/hour.

The baseline repeat is **not bit-identical** to the historical run despite the same seed/recipe: maximum retained parameter difference .00840; historical selected development loss .67565 versus .67653 now. CPU/environment numerical behavior and discontinuous hard-max instance selection are plausible contributors, not isolated causes. The three new arms ran in the same process. Old frozen artifacts were preserved.

## What remains unresolved

A later [frequency coverage check](frequency-coverage.json) found a separate structural issue: frequency stride4 with kernel3 skips **57/256** input rows in the selected CNN and **420/1933** in the full-resolution CNN. An all-positive linear dependency probe establishes missing computational paths independently of learned weights/ReLUs; replacing every pixel in those rows of a real spectrogram changes the trained scores by exactly0. The extra late convolution does not recover the skipped inputs. This measures missing tensor paths, not a percentage of physical sound information or the field-error contribution. See [competing explanations and proposed controls](../../docs/cnn-next-steps.md) and the [quieter-transition diagnostic](../level-transition-v1/RESULTS.md).

The checks exclude a cache mismatch and inability to optimize any real examples. They establish an amplitude-rule cancellation, a concrete compact-architecture invariance, and serious collection/label confounding. They do **not** identify one sufficient root cause for the real-world failures.

The current formulation asks a randomly sampled positive recording sub-bag to contain evidence even though audibility was never annotated. Its max-positive loss routes a positive gradient through only one selected window. The full task also asks a small local network to generalize across microphones, backgrounds and very different positive recording structures after a short training run. The small memorization control removes several of those difficulties at once; its success cannot isolate which one breaks generalization.

The next useful simplification is same-setup audible-on/off examples with temporal labels and uncertain intervals, followed by explicit harmonic/trajectory/tonal-confounder checks. Work from those examples toward broader source generalization. A pretrained CNN such as YAMNet or EfficientNet may help, but improvement alone does not explain the mechanism.

Evidence: [input audit](input-audit.json), [architecture control](architecture-control.json), [memorization control](memorization-control.json), [normalization plan](normalization-plan.json), [per-record comparison](normalization-comparison.json). Scripts are `diagnose_cnn.py`, `diagnose_normalization.py` and `summarize_cnn_diagnosis.py`. New normalization bundles remain trained diagnostics under ignored `runs/cnn-diagnosis-v1/normalization-models/`; none has frozen alarm thresholds or appears in the interface catalog. Byte-identical copies and a checksummed source snapshot are preserved in the [diagnostic packet](../../artifacts/cnn-diagnosis-v1/README.md).
