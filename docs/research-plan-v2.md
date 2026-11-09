# Data-first plan for a useful acoustic drone detector

Status: proposed next-phase protocol. The current detector results are
unacceptable. The existing corpus and all inspected assessments are development
material. Increased positive detections with widespread background alarms do not
constitute detector improvement. Existing software audits remain useful, but
another small same-session experiment is not the next milestone.

## 1. Define the task and freeze the evaluation contract

Detect acoustic evidence of a drone in continuous single-microphone field audio,
including changing flight and competing sounds. Evaluate the complete causal
pipeline, including context, threshold, confirmation, alarm termination and
merging. Do not use any-hit-over-sampled-windows file classification as the
primary success measure.

Proposed engineering targets, to be fixed before training and test use:

- At least 90% event recall on independently annotated audible drone events.
- At most 0.1 false alarm episodes per background hour.
- At most 0.1% of drone-absent time under alarm. Episode counting alone can be
  gamed by one continuously active alarm; occupancy must also be measured.
- A proposed 95th-percentile detection-delay target of 3 seconds after an
  annotated clearly audible onset. Report the entire latency distribution and
  misses, not latency only on an undisclosed easy subset.

These are proposed targets, not achieved results or a guarantee of feasibility.
Audibility, uncertain regions, event matching, hangover tolerance and alarm
merging must be defined before evaluation. Detection of a physically present but
acoustically unobservable drone is a separate claim. Report difficult and
uncertain examples separately without using model outputs to decide audibility.

The blind evaluation budget should include at least 100 distinct drone events
and 30 hours of real drone-absent field audio across multiple capture groups.
Those counts are initial screening requirements, not automatic statistical
proof of the targets. Report confidence intervals and session/site variation;
collect more evaluation exposure when uncertainty is too wide. Do not treat
overlapping windows as independent observations.

**Exit artifact:** a versioned evaluation protocol and an explicit label and
coverage specification. Thresholds are selected on development data, then fixed
globally for held-out conditions. A different threshold per test source is not
evidence of generalization.

## 2. Establish a large, varied corpus before another training study

Initial acquisition targets for the whole corpus, including reserved evaluation:

- At least 100 hours of unique real field timelines, aiming for at least 20 hours
  with supported drone-present intervals and 80 hours of background exposure.
- At least 50 documented capture sessions, 10 drone models, 5 distinct sites and
  3 recording device types, with explicit measures of concentration by campaign.
- At least half of the accepted drone-present time should involve moving flight
  or changing flight state. Limit any one campaign/source to 25% of accepted
  training hours and any one session to 10% of positive training time. Report
  missing strata; dozens of tiny excerpts cannot satisfy session diversity.
- Genuine approach/recede, passes, turns, throttle/load changes, takeoff/landing
  and intermittent audibility, alongside hovering. Include FPV and conventional
  multirotors and hard negatives such as fans, engines, aircraft and tools.
- Both classes under comparable capture conditions in the principal field
  collections. Positive-only collections remain separately identified auxiliary
  material. Synthetic mixtures are separately accounted training augmentation.

These are proposed acquisition requirements, not verified available quantities
or proof that exactly 100 hours is sufficient. More clips cannot compensate for
missing conditions. The feasibility inventory must say whether these requirements
can actually be met and what access, collection, annotation and resources cost.
An unmet requirement is reported as a gap; it is not quietly filled with extra
crops or replayed/synthesized audio.

Count original elapsed time once across channels, overlapping excerpts,
resampling, augmentations and mirrors. Record original capture/flight IDs,
dates, sites, drone identities, flight behavior, devices, label strength, terms,
and links between derivatives. Retain native source precision and keep
transformations reversible. Unknown provenance does not become an independent
recording merely because it has another filename.

Use publisher human annotations where available. Presence somewhere in a long
recording does not label each crop positive. A dominant non-drone sound does not
certify absence. Train on supported intervals/whole labelled inputs; use an
explicitly justified recording-level objective for weak bags or exclude them
from the first supervised comparison. Final event/negative-time evaluation
requires annotations that support those claims. If public labels are insufficient,
the inventory must include a concrete data-access or independent annotation/
collection proposal, rather than assigning invented labels or asking the user
to label another batch.

### Acquisition routes and verified constraints

- Audit original AuDroK campaigns and source recordings, DroneNoise Database,
  IDMT Berne and additional independently captured ground-audio releases.
  AuDroK and large Hugging Face/Kaggle aggregations can contain sources already
  used here. Audit parent identities before assigning any fresh-test status.
- [DroneNoise Database](https://doi.org/10.17866/rd.salford.22133411) is a useful
  overflight candidate. Its primary metadata describes one field campaign at
  Edzell on 17 August 2022; 175 files do not mean 175 independent sessions.
- [The Polish UAV/speech release](https://zenodo.org/records/15190811) has about
  12 GB of files. Its datasheet describes 17 UAVs, four locations and multiple
  dates, but the acquisition uses hovering at fixed measurement positions.
  It can add vehicle/hardware variety, not fill the dynamic-flight requirement.
- Include substantial real background and confuser recordings, with label
  limitations recorded. General sound corpora can supplement training; their
  class names do not automatically provide verified drone-absent test hours.
- Ground listening, onboard ego-noise, laboratory propeller rigs, loudspeaker
  replay and synthetic mixtures have different roles. Keep their origin visible;
  onboard/fault data cannot fill the field-detector coverage requirement.

No current access to the Fraunhofer paper's 71 hours of in-house drone audio and
380 hours of background audio has been established. Nor has access to the
Vilhurin paper's original corpus been established. Public evaluation sets or
third-party reproduction repositories do not provide those training data.

**Exit artifact:** an acquired-data inventory and coverage matrix with measured
unique hours, verified groups, label quality, overlap, actual access and gaps;
plus a frozen acquisition/split manifest. If the public material is insufficient,
produce the specific acquisition/collection budget before further model work.

## 3. Reserve genuinely unseen evaluation before development

All previously scored sources remain development evidence. Reserve new whole
sessions/campaigns and, where feasible, entire sites, device types and drone
types before fitting. Related channels, crops and derivatives stay together.
Inspect metadata/annotation quality without generating detector scores on the
blind set. Trace possible AudioSet/pretraining overlap and disclose what cannot
be ruled out.

Provide a separate development/calibration partition. Use it for model selection,
thresholds, stopping, hard-negative mining and hypothesis-driven error analysis.
The final test is not used for any of those decisions. A failed final test is
then consumed evidence; subsequent changes do not get another independent claim
from repeating it.

The evaluation matrix should distinguish unseen flight sessions, drone models,
devices, locations and dynamic mixtures. Attribute a failure to one factor only
where the acquisition actually separates that factor. Otherwise report a combined
condition change and acquire the missing comparison.

## 4. Establish a bounded, literature-based baseline comparison

Run two bounded preflight checks alongside corpus preparation, motivated by the
user's screenshots. They do not replace acquiring the larger corpus:

- **Gain stability:** score paired real drone and confuser inputs under uniform
  gain changes without clipping/quantization and above the declared silence
  floor. Separately test within-recording level steps/ramps with the causal
  context rule fixed. Keep drone/background ratio fixed for these controls;
  changing only the drone component is a separate SNR test. Retain raw logits
  and decisions so sigmoid saturation cannot make instability appear flat.
  Constant high output fails the negative controls, even if volume invariant.
- **Boundary dependence:** compare actual native first/deeper-layer maps and raw
  signed attribution magnitudes with the renderer. The current CNN uses zero
  `same` padding, and Grad-CAM is ReLU-clipped and maximum-scaled separately for
  each target. Thus bright boundary bands could involve padding, real spectral
  content or attribution behavior; the image alone does not establish which.
  Compare boundary-activation interventions with matched-size interior controls
  and measure raw-logit changes. Such interventions are numerical diagnostics,
  not physical audio trials. Preserve the exact screenshot input/export when
  available; checks on other recordings are not reproductions of that example.

Start with a capable published audio architecture and actual available weights.
[EfficientAT](https://github.com/fschmid56/EfficientAT) releases AudioSet-trained
CNN weights and downstream training code. A fixed MobileNet configuration such
as `mn10_as` is a concrete public baseline candidate. Pin its code, checkpoint,
frontend, precision, parameter count and initialization before the comparison.

Run the following on the same corpus, labels and splits:

1. A credible non-neural classifier on spectral summary features, with an
   energy/activity-only control to test the hypothesis that the system detects
   generic sound rather than drone-specific evidence.
2. The chosen audio-pretrained CNN, fine-tuned for this task.
3. Exactly the same CNN architecture and frontend initialized from scratch,
   with sufficient training and an equal declared validation-tuning budget.
4. The pretrained CNN with the paper-motivated augmentation package: real
   background mixing, microphone response variation, drone pitch variation and
   limited spectral masking. Mix/pitch only where source/label structure supports
   the operation; label composites explicitly and count no synthetic hours as
   independent real recordings.

Use three seeds for each neural condition. Verify random initialization does not
silently load ImageNet/AudioSet parameters. Compare learning curves and ensure
the scratch model is not simply stopped before convergence. Retain the existing
detectors as fixed references, without generating another broad architecture or
normalization sweep.

This is a public-data comparison informed by
[Reuter et al.](https://arxiv.org/html/2605.31329v1), not an exact reproduction of
their private-data experiment. Their SE-ResNet and training corpus are different
from EfficientAT. Any deviation from an author's actual implementation must be
explicit. PCEN/attention from
[Vilhurin et al.](https://arxiv.org/html/2608.14287v1) are specific subsequent
hypotheses, not assumed improvements.

**Exit artifact:** a reproducible baseline table at the same false-alarm budget,
with per-condition errors and learning curves. If a baseline fails easy,
well-labelled development contrasts, investigate implementation/labels before
calling the result a generalization limit.

## 5. Make further experiments conditional on evidence

Every follow-up must specify the observed failure, a testable explanation, the
one change being tested, the fixed controls and the development metric required
to retain it. For example, test PCEN against persistent background/gain effects,
or temporal pooling against intermittent detection failures, only after the
corresponding error is demonstrated. Inspect actual audio/tensors and matched
confusers to check physical explanations. Grad-CAM and perturbations remain
numerical sensitivity evidence, not proof of acoustic causality.

Retain a change only if it improves detection at the fixed false-alarm/occupancy
budget across relevant held-out development groups. A positive-count increase
with worse background response fails this rule. Fine-resolution or richer-input
experiments follow a specific representation failure; they are not prerequisites
for this first strong baseline. Multi-view/3D ideas remain deferred.

Select the candidate using development evidence, freeze its whole alarm policy,
then perform one blind continuous evaluation. Report misses, false episodes,
false-alarm duration, latency, uncertainty and worst-condition behavior. A result
that misses the agreed targets is a failed result and is reported that way.

## 6. Resource and execution order

Use [the durable storage and recovery design](storage-and-training-v2.md) for
the selected free Colab GPU notebook with Google Drive, local training-cache
operation and a complete-checkpoint resume path. Code lives in the repository;
Drive holds experiment state across notebook replacements. Runtime verification
precedes an actual audio throughput pilot and integration of the selected CNN.

The Codex workspace has limited free disk and no exposed NVIDIA GPU; it is not
the bulk-storage/training host for the proposed corpus. A Colab GPU allocation
still needs to be verified in the notebook. For scale, 100 hours of
48kHz mono PCM needs about 35 GB at 16-bit or 69 GB at 32-bit before compression,
with substantially more space for multichannel originals and derived data.

Budget durable storage (initially 100–200GB, revised from the inventory), a
GPU training environment, and annotation/access needs before bulk acquisition.
Keep compact waveform caches and compute spectra in batches instead of caching
every overlapping tensor. Benchmark throughput before giving a training-time
or cost estimate. Do not claim unavailable resources are already provisioned.

The immediate work order is: task/evaluation requirements plus verified
data/access/coverage and resource inventory; acquisition and blind split;
frozen executable training/evaluation protocol; the bounded baseline comparison;
one justified follow-up if needed; blind continuous evaluation. Case inspection
supports this sequence and does not replace obtaining sufficient varied data.
