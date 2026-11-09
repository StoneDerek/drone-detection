# What was built, trained and measured

The assessed default is a new, reproducible audio-transfer-learning baseline and
a local recording-review application. It uses the official frozen Google YAMNet
v1 encoder, based on MobileNet-v1 and pretrained on AudioSet. It is separate from
the historical EfficientNet implementation. The product still has material
generalization failures; a field-reliable detector has not been established.

## Default architecture and input

Audio file → float waveform → arithmetic mono downmix → 16 kHz resampling →
3-second rolling context → YAMNet's own 64-band log-mel frontend and frozen
MobileNet → mean of the 1,024-dimensional patch embeddings → train-derived
standardization → one logistic-regression output → alarm hysteresis.

The classifier fits 1,024 weights and one bias. The 1,024 feature means and
scales also come only from training data. YAMNet stays frozen. Its original
AudioSet class scores are not used for the drone decision. This task-specific
training is a small classifier fit, not end-to-end neural-network fine-tuning.

The generic configuration's `mel_bins: 48` belongs to the hand-engineered
baseline. It is unused when `representation` is `yamnet`; the frozen encoder's
own 64-band frontend, recorded under `encoder.input_spec`, is authoritative.

Each full context is 48,000 mono samples. A score is produced every 0.5 seconds.
The default alarm requires two consecutive scores at least 0.5; it clears after
two below 0.375. Silence, short input and one possible padded final context are
explicit. File inference and recorded-stream replay share this engine. There is
no claimed live microphone capture or physical acoustic range.

## Actual selected data

| Split | Real files | Positive files | Negative files | Available audio hours |
| --- | ---: | ---: | ---: | ---: |
| Training | 405 | 55 | 350 | 1.912 |
| Development | 210 | 22 | 188 | 1.908 |
| Original frozen v2 assessment | 1,741 | 133 | 1,608 | 3.380 |
| Total | 2,356 | 210 | 2,146 | 7.200 |

Training comprises D3Set from its author-linked Kaggle mirror (10 positive
ground recordings and 50 negative recordings), Svanstrom (30 positive and 60
negative clips), DDL real field audio (15 positive excerpts from five flights),
and ESC-10 (240 negatives). Development additionally uses earlier Batear field
audio. New-day Batear, another DDL day and unused ESC source IDs/categories formed
the original v2 assessment. Dataset/member hashes and conservative source/day
groups are retained. Raw archives are larger than this selected corpus.

Training samples at most 60 uniformly distributed contexts per recording,
rather than exhaustively using every overlapping context. The default used
**4,648 sampled contexts: 1,262 from positive bags and 3,386 from negative bags**.
These are overlapping inputs, not independent recordings. The full training
recordings total 1.912 hours; the sampled contexts are not an additional 3.9
independent hours.

## Training procedure and outcome

Labels come from publishers at recording level. A positive file can contain
intervals without audible drone sound. The multiple-instance procedure starts
with a classifier on each recording's mean embedding, then performs four rounds
of selecting the highest-scoring context from each positive recording and
refitting against all sampled negative contexts. Recording mass and class mass
are balanced. The classifier uses L2-regularized logistic regression, C=1,
L-BFGS, seed 42. No made-up event boundaries are assigned to positive files.

Default feature extraction and classifier training took 147.94 seconds on the
available Xeon CPU; development scoring took 302.60 seconds. Decision thresholds
were selected on development data. Development found 20/22 positive files with
9/188 negative files alarmed. The frozen assessment found 100/133 positives and
alarmed on 126/1,608 negatives: **75.2% recording recall, 7.84% negative-file FPR**.
Both provisional recording targets failed.

The fully non-neural relative log-mel summary plus logistic-regression baseline
found 67/133 positives and alarmed on 109/1,608 negatives on that same assessment.
This provides an actual neural versus non-neural comparison, albeit with
different representations. The source results vary substantially: YAMNet found
81/90 DDL excerpts but only 19/43 Batear positive clips. Those positive assessment
files cover only two held-out capture-day groups. Weak labels and correlated
captures prevent treating the file count as independent field trials.

An initial real-data run failed much more severely and remains preserved. Its
failure informed broader positive coverage and revised threshold selection in
v2. The later gain-normalization experiment is development only: 21/22 positives
and 9/188 negatives alarmed. It did not replace the assessed default or supply a
fresh reliability benchmark. Synthetic signals are software tests only.

## Supplied historical repository

The newly supplied [ShikangHu/dronedetection repository](https://github.com/ShikangHu/dronedetection/tree/c1e44605c07004501b72e4c3dc0c08c3af3f3f01)
is accessible at June 9, 2026 commit `c1e44605c07004501b72e4c3dc0c08c3af3f3f01`.
It contains EfficientNet-B0 code, a 16 MB `pretrained/best_model_v3.pt`, real
`mean.npy`/`std.npy` files and split CSVs. That makes it a recoverable historical
baseline. Its checkpoint has not yet been run in this environment.

The actual code takes mono 22,050 Hz audio in 3-second windows with a 1.5-second
hop, computes 128-band log-mel spectrograms, repeats the single input channel to
three channels, and uses ImageNet EfficientNet-B0 → 1,280 features → dropout 0.3
→ one linear output. Its file script uses majority voting over window decisions.
The config specifies five head-only epochs, subsequent whole-network fine-tuning,
AdamW and up to 50 epochs. The reported epoch-25 result comes from the project
summary; it has not been independently reproduced here.

Counting the supplied CSV rows confirms 69,727 training files (25,604 positive),
14,948 validation files and 14,903 test files. The often-quoted 99,578 files is
the total across all three splits. CSV row counts do not verify raw-audio
availability, independent originals or the checkpoint's training history.
There are no exact path duplicates across the three CSVs; derivative/session
overlap requires more than a path check. The default and historical scores come
from different evaluation protocols and cannot establish which model is better
without an overlap-audited comparison.

## Current spectrogram experiments

The new lab tests actual tensor dimensions rather than enlarged screenshots:
linear versus mel frequency axes, 1,024 versus 4,096-sample Hann analysis
windows, 64 versus 256 mel bands, and log versus linear relative power. All use
the same compact 2D CNN recipe. A separate RBF SVM uses per-frequency temporal
means/std without a neural encoder. These are development experiments on the
same training/development files. The assessed default and original results are
preserved.

Two diagnostic follow-ups address confounding in the initial sweep: a 256-band
uniform-Hz filterbank matches the 256-mel tensor to separate band placement from
input size; a frequency-only convolution expands the high-resolution CNN's
receptive field from about 98 Hz to 598 Hz. Both remain explicitly development
work, with separate plans and preserved earlier results.

The lab exposes the audio, score timelines, each model's input tensor and genuine
Grad-CAM from selectable CNN convolution layers. Grad-CAM shows sensitivity of a
selected logit, which can help identify shortcuts; it does not prove causal or
useful recognition. No Grad-CAM is fabricated for the SVM.

## CNN foundations follow-up

The [new controlled experiments](../reports/cnn-foundations-v1/RESULTS.md) keep the
same405 training/210 development files and256×172 uniform-Hz log-power input.
They repair missing frequency paths, test a93,313-parameter network and compare
nearly equal154,675/154,689-parameter global versus frequency-preserving readouts.
This remains one spectrogram per context; no3D or multi-view input is used.

The old scratch-CNN bag objective had a demonstrable noise loophole: positive
maximum loss and negative average loss can reward input-independent dropout.
The repaired pair uses BCE of the same maximum context logit for either class,
dropout0, balanced4+4 recording batches, sample8 per recording, Adam.001,
L2.0001 and24 epochs. Checkpoints minimize deterministic consistent-max
development BCE. The frequency readout is also repeated with seed7, with its
interim development-triggered selection disclosed. This finding concerns our
scratch-CNN implementation; the linked historical EfficientNet code uses a
different training pipeline. No historical failure is attributed to this loss
without evidence.

The report preserves failed controls, actual training exposure, source-level
errors and full dense replay calibration. New browser comparisons do not replace
the separately assessed YAMNet product default or create a fresh test set.

## Subsequent training and preprocessing audit

The [audit](../reports/training-audit-v1/RESULTS.md) independently checks the actual
waveforms, FFTs, caches, loss gradients, Adam updates and checkpoint arrays. It
finds that assigning full-recording labels to sampled sub-bags changes the intended
weak-label objective. The optional full-recording selection path scans all cached
windows at every update and recomputes the maxima with gradients. Its one-seed
control detects 14/22 development positives, versus 16/22 for the preserved sampled
baseline, at the same 9/188 calibrated negative-file count. The correction is not
promoted and does not establish an accuracy gain.

Original-source verification also finds 15 DDL training files unnecessarily reduced
from 32-bit to 16-bit PCM. Original-precision float WAVs and a separate train/development
manifest are recovered under ignored `runs/training-audit-v1/`; the original corpus
and all frozen models remain unchanged. The precision variant was not used for the
sampling control. No fresh assessment or new audible labels are introduced.

## Formulation and normalization diagnosis

[Specific real recordings](../reports/real-observations-v1/OBSERVATIONS.md) include
harmonic drone, helicopter and tonal-background examples, and a largely broadband
flight-labelled excerpt. They constrain the next formulation before another
architecture search. No long-context or multi-view branch was added.

The [controlled diagnosis](../reports/cnn-diagnosis-v1/RESULTS.md) verified all
7,299 cached train/development tensors exactly and demonstrated small-real-set
memorization. Upstream scalar RMS gain cancels under the existing per-context
spectral peak normalization. New diagnostic-only arms use an explicit fixed
STFT power reference, with and without the .05 RMS target and gain/peak caps.
Those arms change inputs and optimization but have no frozen alarms, no new
independent assessment and no position in the interface catalog. Existing model
identities, preprocessing defaults and decisions remain the same.

The [Cloudflare browser lab](https://drone-audio-lab.pages.dev) exports existing
weights to numeric arrays and uses the official Google TFJS YAMNet encoder. Its
port was checked against Python on 176 context/model scores from three existing
real train/development files, six Grad-CAM target/layer cases and decoding at
16/44.1 kHz. Numerical parity is software evidence, not 176 detection trials.

## Future input-space idea: 3D CNN

Parked research idea from the user: useful acoustic functions or transforms could
define an input volume for a 3D CNN. Work through specific real examples first to
identify the distinctions that the representation should preserve. Define what
the third axis represents and whether neighboring slices have useful local
relationships for convolution; an arbitrary stack of unrelated transforms does
not automatically supply that structure. This is a future hypothesis, with no
implementation or training requested now. No particular transform or surrounding
context window has been chosen.
