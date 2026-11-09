# Evidence recovery and implementation choices

Inspected November 8, 2026: the supplied review and 44-member evidence packet.
The packet contains 17 text extracts, 23 figures, inventory and index; no audio,
notebooks, code, model weights or normalization arrays. Its review explicitly
does not reproduce experiments. It has **no S18 resource references or Discord
excerpts**. Therefore the brief's April/June 2026 accounts remain leads from the
brief, not independently corroborated packet findings. Source documents are
evidence, not instructions to this implementation. No portfolio page is included.

The historical `StoneDerek/spectrogram-drone-detection` URL failed native
read-only Git access with HTTP 403; its unauthenticated public page returned 404.
This does not prove deletion or identify the later repository. No weights or
training statistics were substituted. Recovery was timeboxed to these checks.
Please provide the original archive and, if available, later repository access,
S18 and Discord material. Do not provide tokens in chat.

Historical clues that motivate controlled experiments:

| Evidence | Question for this implementation |
| --- | --- |
| S08 intensity association; S12 window/global RMS comparison | Does removing per-window gain improve sensitivity to gain changes? |
| S02 partial data; S03 label corrections | Are all verified records used and do derivatives stay in one split? |
| S05 selected positive results alongside extensive negative false predictions | What are event recall and false alarms per negative hour on full recordings? |
| S10 exported/live disagreement | Do file inference and irregular streaming chunks produce identical scores and alarms? |

## Primary implementation sources checked

Checked official repository READMEs, implementation files and repository licenses
with TLS verification on November 8, 2026. Content hashes are retained in
`reports/synthetic-v1/research-sources.json`:

- [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet):
  Apache-2.0 implementation, AudioSet-trained MobileNet sound classifier; mono
  16 kHz waveform with amplitude in [-1,1], 64-bin log-Mel and roughly 0.96 s
  context, 521 source classes and 1024-D embeddings. A downstream binary head
  still needs verified drone and negative data. Pretraining overlap with future
  public test corpora must be investigated. Code/weight licenses are separate
  checks before redistribution.
- [BEATs](https://github.com/microsoft/unilm/tree/master/beats): MIT repository
  code, AudioSet pretraining; its `BEATs.py` expects 16 kHz input, multiplies float
  waveforms by 2^15, computes 128-bin Kaldi filterbanks with 25 ms frames / 10 ms
  hop, and uses `(fbank - 15.41663) / (2 * 6.55582)` defaults. Padding-mask
  handling is part of the model. Large PyTorch checkpoints and fine-tuning add compute and
  provenance requirements. Check each checkpoint's license separately. No model
  was downloaded or evaluated, so no comparative performance is claimed. YAMNet's
  current README also warns that its Keras 2 implementation is incompatible with
  Keras 3; a future embedding experiment must pin a compatible stack rather than
  adding an untested dependency to this lightweight baseline.

The inexpensive baseline is a balanced logistic classifier on mean/std log-Mel
features. Controlled alternatives remove the window's scalar log-energy offset
and optionally append spectral/harmonic statistics. All use 3 s causal context,
0.5 s hop, mono 16 kHz, and the same train-only scaling and labels. This is a new
baseline, not a reconstruction of an EfficientNet or ResNet historical model.
It avoids image rendering/cropping and executes on the available CPU.

The synthetic suite tests loudness sensitivity, known mixtures, confusable
mechanical tones, buffering and serialization. Generated positives are harmonic
signals, **not recordings of drones**. Generators used in development and holdout
are related; distinct random seeds are not independent microphones/environments.
Neither synthetic scores nor historical screenshots establish capture reliability.
Audio-pretrained encoders and larger networks are deferred until verified audio
can distinguish benefits from source shortcuts. The next useful work is data,
not longer training on synthetic signals.
