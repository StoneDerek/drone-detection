# Data-first plan for a useful acoustic drone detector

Status: proposed next-phase protocol. The current detector results are
unacceptable. The existing corpus and all inspected assessments are development
material. Increased positive detections with widespread background alarms do not
constitute detector improvement. Existing software audits remain useful, but
another small same-session experiment is not the next milestone.

A [broader original-source audit](../reports/source-scope-audit-v3/README.md)
now covers all ten sources identified by the user, alongside the earlier field
and background candidates. Source availability, acquisition and suitability for
a particular experiment are separate statuses. There is no 100-hour readiness
gate; that earlier number was an unvalidated planning estimate.

The [completed outdoor intake audit](../reports/outdoor-source-audit-v1/README.md)
adds about 2.94 point-microphone file-hours from one campaign. The much larger
recorder-view count includes averaged arrays. Matched negative labels and
independent capture diversity remain open requirements. This subset does not
establish either the size of the entire AuDroK collection or a public-data shortage.

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

Size the blind evaluation separately from training, using the required precision
and condition coverage. For example, with zero false alarm episodes in T hours,
the one-sided 95% Poisson upper rate is -log(0.05)/T, approximately 3/T. About
30 hours with zero episodes would only put that bound near 0.1/hour under the
Poisson assumptions; it would not establish performance in unrepresented
conditions. Clustering by session, nonstationarity and observed alarms change
the required exposure. Report session-aware uncertainty and condition breakdowns.
Likewise, observing 90 detections among 100 independent events gives an estimated
90% recall, with a 95% Wilson interval of approximately 82.6–94.5%, rather than
proof of at least 90% recall. Related events require grouped uncertainty estimates.
The earlier 100-event/30-hour suggestion is not a universal minimum or sufficient
evaluation design. Do not treat overlapping windows as independent observations.

**Exit artifact:** a versioned evaluation protocol and an explicit label and
coverage specification. Thresholds are selected on development data, then fixed
globally for held-out conditions. A different threshold per test source is not
evidence of generalization.

## 2. Establish a large, varied corpus before another training study

Determine the needed quantity from coverage and grouped learning curves. The
previous 100-hour total, 20/80-hour class allocation, 50-session count, model/site/
device counts, dynamic-flight fraction and concentration caps were proposals
without a task-specific sample-size analysis. They are withdrawn as mandatory
requirements. A fixed number of hours cannot certify generalization, and no
current result establishes the amount needed for this task.

Acquire broadly from the identified original sources while recording:

- Supported drone-present time, background time and unknown/unlabelled time;
  file duration is not automatically positive duration.
- Original capture groups, campaigns, sites, dates, drone models and recording
  devices. Report their distributions and missing metadata, not just totals.
- Moving flight and changing flight state, alongside static conditions. Measure
  concentration by campaign and session so extra windows from one recording do
  not masquerade as added coverage.
- Genuine approach/recede, passes, turns, throttle/load changes, takeoff/landing
  and intermittent audibility, alongside hovering. Include FPV and conventional
  multirotors and hard negatives such as fans, engines, aircraft and tools.
- Both classes under comparable capture conditions in the principal field
  collections. Positive-only collections remain separately identified auxiliary
  material. Synthetic mixtures are separately accounted training augmentation.

These are coverage dimensions, not a claim that all combinations are necessary
or publicly available. Inspect the original releases before deciding which gaps
require new collection. DADS membership is a source lead, not a quality rating.
Do not assume a public-data shortage from a small audited subset, and do not
fill a missing condition with extra crops or replayed/synthesized audio.

### Estimate training-data needs empirically

After broad intake and source grouping, reserve the blind evaluation and a
separate, varied development set. Form nested training subsets by adding whole
capture groups, with class/condition coverage tracked at each size. Where there
are enough groups, use several predeclared sizes, such as 25%, 50% and 100% of
available groups, and repeat group selection/training seeds to expose selection
variance. Percentages are study design choices, not readiness thresholds.

Use the same subsets and development conditions for the neural/non-neural
comparison below. Give each fit enough optimization to converge and select its
global alarm threshold on development data. Plot event recall, false episodes,
alarm occupancy and delay against both independent group count and unique time,
including per-condition errors and uncertainty. Do not repeatedly inspect the
blind set to construct these curves.

Keep acquiring when extra independent groups improve held-out development
performance or fill a demonstrated coverage gap. A flat curve with unacceptable
errors triggers investigation of labels, source/class confounding, optimization
and missing acoustic information; it does not prove that more of the same data
will help. Stop scaling for the current study when its predeclared development
targets and precision are met and additional groups have little material benefit,
then freeze the candidate for the separate blind evaluation. Until these curves
exist, report the required training quantity as unknown, not as 100 hours.

Count original elapsed time once across channels, overlapping excerpts,
resampling, augmentations and mirrors. Record original capture/flight IDs,
dates, sites, drone identities, flight behavior, devices, label strength, terms,
and links between derivatives. Retain native source precision and keep
transformations reversible. Unknown provenance does not become an independent
recording merely because it has another filename.

Use publisher human annotations where available. Their supplied class labels
can support auxiliary training without requiring the user to relabel every
second. Record label strength and check representative samples. Presence
somewhere in a long recording does not label each crop positive; a dominant
non-drone sound does not certify continuous drone absence. Train on supported
intervals/whole labelled inputs; use an
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
plus a frozen acquisition/split manifest. If the completed source audit exposes
a specific unmet coverage or label need, produce the corresponding acquisition/
annotation proposal. Do not declare a shortage solely because an arbitrary hour
target has not been reached.

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
still needs to be verified in the notebook. For storage planning, each hour of
48kHz mono PCM needs about 0.346 GB at 16-bit or 0.691 GB at 32-bit before compression,
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
