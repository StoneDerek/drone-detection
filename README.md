# Acoustic drone detection

A product for reviewing timestamped drone-presence scores from audio,
playing/seeking the recording, inspecting alarm intervals, and exporting results.
It includes real-audio training and reproducible source-aware evaluation.

The public [browser model lab](https://drone-audio-lab.pages.dev) runs the frozen
CNNs, non-neural SVM and YAMNet control on your device, with audio playback, score
timelines, Grad-CAM and five playable real examples. Your selected audio stays in
the browser. WAV input is supported; the Python interface below also supports
FLAC/OGG. Draft interval listening notes can be downloaded with the audio hash.

All five examples load together; additional uploads and their results stay available
in a shared comparison table. Grad-CAM can inspect any of the spectrogram
CNNs, preserving the selected interval. Both plots support linked zoom/pan,
heatmap or overlay display, adjustable opacity and native attribution cells.

The [seven real case studies](reports/real-observations-v1/OBSERVATIONS.md) and
[CNN diagnosis](reports/cnn-diagnosis-v1/RESULTS.md) distinguish visible acoustic
structure, representation loss, weak labels and training behavior. Cache parity,
small-set memorization and controlled normalization checks are diagnostics, not
new field-accuracy results. See [browser build/deployment](docs/cloudflare.md).

The [CNN foundations experiments](reports/cnn-foundations-v1/RESULTS.md) identify
a concrete positive-max/negative-mean training flaw: dropout noise can improve
the objective without useful input information. Corrected training and matched
154,675/154,689-parameter readout controls use the same real audio and frontend.
The report retains unsuccessful controls, source errors and the second seed.
The new models are available in the browser lab; results remain development evidence.

The subsequent [training and preprocessing audit](reports/training-audit-v1/RESULTS.md)
checks independent FFT/decoder references, original source precision, gradients,
Adam updates, checkpoint identities and replay padding. It identifies a remaining
sampled-window target mismatch and unnecessary DDL bit-depth reduction. Isolated
corrections and an unsuccessful same-data training control are preserved separately.

The [transfer case diagnosis](reports/transfer-diagnosis-v1/RESULTS.md) separates
threshold failures from learned input sensitivities using actual recordings,
temporal probes and disclosed band occlusions. A
[simplified learning study](reports/learnability-normalization-v1/RESULTS.md)
shows the repaired scratch CNN can learn the published one-second labels at two
seeds, with matched non-neural controls and peak, fixed, bounded-RMS and exact-RMS
amplitude rules. This is consumed single-session diagnosis. It does not establish
cross-source reliability or that harmonic motion was learned. The
[current priorities](docs/current-research-priorities.md) preserve the user's
request to understand the relevant distinctions before adding richer inputs.

The [cross-condition checks](reports/cross-condition-v1/RESULTS.md) carry those
same twelve packages into other real recordings without refitting. They retain
positive-only human-labelled Kenya clips, explicit field/replay backgrounds from
a separate controlled source, and every source's false positives. An exact
frequency-contribution account shows why the matched non-neural control's strong
within-session distinction does not carry over to FPV recordings.

The [model lab](reports/spectral-lab-v1/RESULTS.md) compares trained spectrogram
CNNs, a fully non-neural SVM, and the existing YAMNet/logistic controls. It shows
the input beside genuine Grad-CAM, with context/layer/class selection, frequency
zoom, training curves and JSON export. Run `bash scripts/start_lab.sh` after
setup. The [architecture and data account](docs/model-and-data.md) explains
exactly what was trained, how much data was used and how the historical
EfficientNet repository differs. The lab is development work; the assessed
default launched below is preserved.

**Research prototype:** publisher recording labels are weak supervision. Field
reliability, event recall, false alarms per hour, range and live capture are not
established. The [real results and failures](reports/real-v2/RESULTS.md) state what
passed and failed; the interface uses “No alarm” rather than claiming absence.

## Run the product

Python 3.12 and uv are required. From this checkout:

```bash
bash scripts/setup.sh
bash scripts/start.sh
```

The server prints its local address. Upload WAV, FLAC or OGG audio (up to 64 MB and
10 minutes), select **Analyze recording**, then play or seek the timeline. Download
scores as JSON. When the evaluation audio is present, the examples include fixed
positive, ordinary-negative and mechanical-negative inputs, plus a clearly labeled
observed-failure appendix. Upload analysis works without the evaluation corpus.

The timeline also flags clipping and strong low-frequency energy. Listen to the
flagged sections when reviewing a result; these checks do not identify wind or
measure SNR. Quality flags never suppress an alarm. Silent and insufficient
audio have their own display states. See the [0.2 product package](artifacts/product-v0.2/README.md)
for the installable update and verification evidence.

The model and all required statistics/encoder files are bundled in
`artifacts/real-v2/candidate`. Inference does not download weights or contact an
external service. Uploaded audio is processed locally and its temporary file is
removed after analysis. The app supports recorded replay; it does not advertise
microphone capture or a validated physical distance/SNR range.

## File and recorded-stream commands

```bash
.venv/bin/drone-detector infer --bundle artifacts/real-v2/candidate \
  --audio /path/to/recording.wav --output runs/my-file-scores.jsonl
.venv/bin/drone-detector stream --realtime --bundle artifacts/real-v2/candidate \
  --audio /path/to/recording.wav --output runs/my-replay-scores.jsonl
bash scripts/check.sh
```

File and stream commands share the same preprocessing and alarm engine. They use
mono 16 kHz audio, three-second context and a half-second step; short/padded tails,
silence and insufficient context are explicit. Stereo file input is averaged.
For a microphone array, select one physical channel first, as the dataset scripts
do. Scores are uncalibrated classifier outputs. Each trace identifies the model.

## Real data and experiments

The data comes from the author-linked **Kaggle D3Set** mirror, Svanstrom,
**real-field DDL** on Zenodo, Batear, and ESC environmental recordings. DDL's
synthetic archive is excluded. Original/member hashes, licenses, flight/session
identities, grouped splits and weak-label evidence are recorded in the manifest.
Raw third-party audio remains outside tracked source/artifact files.

The first real assessment failed: the selected detector found only 3/48 positive
files. That assessment was explicitly consumed for development, not recycled as
fresh evidence. Version 2 broadens positive sources, corrects duration-biased
threshold selection, and reserves new field days and negative source IDs.
Read the [brief investigation](docs/real-data-investigation.md) and
[full assessment](reports/real-v2/RESULTS.md), including source breakdowns and errors.
File/window counts are not independent flight or session counts.

Pinned scripts reproduce the data workflow:

```bash
.venv/bin/python scripts/fetch_public_audio.py
# Download D3Set version 1 from the Kaggle URL below into runs/real-v1/downloads/.
.venv/bin/python scripts/prepare_real_audio.py
.venv/bin/python scripts/ddl_audio.py
.venv/bin/python scripts/prepare_real_v2.py
.venv/bin/python scripts/fetch_yamnet.py
.venv/bin/python scripts/train_real_v2.py
```

D3Set v1 download:
`https://www.kaggle.com/api/v1/datasets/download/francoismarcoux1212/d3set-introlab?datasetVersionNumber=1`
Save it as `runs/real-v1/downloads/kaggle-d3set-v1.zip`. The scripts verify the pinned
archive and original Git/LFS content. Source scripts stop rather than substituting
missing statistics or accepting changed bytes. Interrupted v2 preparation supports
`--resume`, which verifies cached downloads. Existing published model packages
are preserved. Initial v1 source/split construction is documented in
[the earlier workflow](docs/legacy-workflow.md); its synthetic measurements are
software checks only.

Training/tuning uses development data only. `scripts/train_real_v2.py --assess`
opens the fresh assessment **once**, after freezing both arms and demonstration
IDs. Its local final-use ledger refuses another assessment on that corpus.
If errors guide further training, keep the set as development and obtain fresh
assessment recordings. Replaying an existing frozen model does not create new
independent evidence.

The supplied acoustic-detection article guides the
[next product experiment](reports/context-gain-v1/RESULTS.md): bounded gain
normalization within each available context, compared on the same real training
and development data. It is an opt-in development experiment; the assessed
version-2 model remains the default. The article's quoted ranges and network
interception rates are not product specifications or classifier evidence.

## Evidence and limits

- [Model/source results, thresholds, durations and failures](reports/real-v2/RESULTS.md)
- [Model package](artifacts/real-v2/README.md)
- [Original event-level contract](contracts/evaluation-v1.json)
- [Manifest and label conventions](data/README.md)
- [Collection/annotation protocol](docs/collection-protocol.md)

The full original project archive is still optional material for future auditing;
it is not a prerequisite to running this product. A field-readiness claim needs
more independent capture days/devices and independently reviewed event boundaries
and negative stretches. The portfolio page and environment publication are separate
work. This task does not publish the environment or deploy the app externally.
