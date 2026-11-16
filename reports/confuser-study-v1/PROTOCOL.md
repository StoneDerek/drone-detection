# Mechanical-confuser follow-up

The preceding study found 1,329 false positives in 8,221 UrbanSound8K crops. Drills,
horns, air conditioning and jackhammers accounted for 1,090 of them. That test is
consumed. This study uses it for development and reserves different public audio
to test whether an improvement transfers.

Three comparisons are specified before fresh detector scores are produced:

| Variant | Head | Cutoff |
|---|---|---|
| Original | Previous trained BEATs linear head | Previous TUT-only cutoff |
| Recalibrated | Same previous head | Broader development negatives |
| Confuser-trained | Retrained linear head with additional UrbanSound8K negatives | Same broader calibration rule |

BEATs itself remains pretrained and frozen. No encoder fine-tuning, new
architecture, frontend change, window sweep or hyperparameter selection enters
this comparison. The original positive TRAIN and TUT/DNC negatives stay in TRAIN.
All accepted UrbanSound8K classes are included, rather than selecting individual
previous mistakes. A deterministic split assigns approximately 75% of Freesound
parent groups to TRAIN, stratified by publisher sound class. Remaining parents
join DEVELOPMENT. Related clips never straddle that partition. Previously
consumed positive test collections become development monitors.

Both heads use weighted logistic regression, C=1, seed42, weighted TRAIN-only
scaling, and equal mass for classes, sources within class, and provenance groups
within source. The total sample-weight mass stays at the original 9,959, keeping
the regularization scale comparable when rows are added. The new head trains on
16,048 crops; DEVELOPMENT contains 4,525. Positive labels do not choose cutoffs.

Broader calibration sets one global raw-logit cutoff: the maximum of the cutoffs
needed to meet an empirical crop FPR of at most1% separately on TUT evaluation
negatives and held-out UrbanSound8K parents. This is a development constraint,
not a promise about unseen data or false alarms per hour. Comparing Original
with Recalibrated isolates the cutoff change. Comparing Recalibrated with
Confuser-trained tests the effect of the new head at the same calibration rule.

The reserved candidate cohort is:

* **FSD50K official evaluation:** 7,955 Freesound recordings from 2,123 uploaders,
  including a prespecified 390-recording mechanical/road-vehicle subset. All
  known UrbanSound8K and ESC-50 parents are excluded. Where FSD metadata links a
  known parent to an uploader, that uploader's evaluation audio is also excluded.
  Human aircraft labels and explicit drone/UAV metadata are excluded. The full
  eligible cohort is used; model predictions do not select difficult cases.
* **WSU DJI Air3:** all 1,486 complete, nonoverlapping three-second grid cells
  supported by publisher alignment and airborne telemetry above one foot,
  across four fixed recorder views. Two March flights share a date/site group;
  there are three date/site groups, not1,486 independent trials. An additional
  190 windows come only from the two explicitly bounded no-drone tail intervals.
  Unknown initial/final spans are excluded. The eight incomplete trailing bytes
  of two WAVs are never read as a frame. Channel0 is a physical microphone.
* **AuDroK Dresden:** one central crop from each of the58 indexed point-microphone
  recordings, covering seven publisher drone codes within **one campaign**.
  Array averages are excluded. Recording-level drone labels remain weak labels
  for the selected crops; this is a secondary stress test.

Native audio is decoded without gain normalization. FSD/AuDroK use one central
crop of up to three seconds; WSU uses the specified grid cells. All crops are
resampled to16kHz float32 before the unchanged native BEATs frontend. Nonfinite
signals fail preparation; zero or shorter-than0.25s crops are excluded. Exact
decoded crop identities are checked within the fresh cohort and against the
previous corpus. These checks cannot rule out all shifted, edited or lossy
copies, unidentified source links, or overlap in BEATs pretraining.

FSD metadata and both audio volumes must match publisher MD5s. AuDroK partial
archive reads check the pinned member index, complete-member CRC and observed
SHA256; they do not independently recheck the whole-archive SHA. WSU reads check
the pinned native prefix, exact byte ranges, source length, last-modified value,
and observed range hashes. No publisher whole-file checksum is available for
those reads. Acquisition receipts preserve these distinctions.

Primary reporting separates source and class. It includes crop recall, crop
FPR, provenance-group averages and paired differences. FSD uncertainty uses a
paired uploader-cluster bootstrap with2,000 draws and seed42. Field results
remain descriptive because campaign diversity is small. No crop-independent
confidence interval, false-alarm-per-hour estimate or detection-delay claim is
made. The labels establish publisher presence or foreground classes; audible
drone onset/offset and exhaustive absence have not been independently annotated.

Every head, cutoff, reserved cohort plan and encoder identity is hashed before
the fresh comparison. The complete cohort is scored in one fixed batch for all
three variants. Its results become consumed development evidence afterward.

Primary sources: [FSD50K](https://zenodo.org/records/4060432),
[WSU publication files](https://rds.westernsydney.edu.au/Centres/ICNS/2026/Jahir_Rodriguez/README.md),
and the pinned AuDroK source index in
[`data/outdoor-point-recordings-v1.jsonl`](../../data/outdoor-point-recordings-v1.jsonl).
FSD per-recording attribution and licenses are retained. Raw audio is not
redistributed in this branch or its result bundle.
