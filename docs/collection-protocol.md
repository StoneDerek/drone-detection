# Collect and annotate the missing evidence

No current microphone, environment, drone, distance or SNR range is validated.
The missing field-validation input is independently reviewed continuous audio
with audible-event boundaries and matched negative sessions across more capture
days/devices. The current public-data pipeline runs without the historical WAV
archive. That archive remains optional material for auditing the earlier project;
its inventory alone cannot establish reliable labels or recording independence.
Retain permission and provenance for additional recordings; dataset and model
licenses are separate.

1. Retain original recorder files, timestamps and channel/rate settings. Give
   each continuous original, session, site, device and drone a stable identifier.
   Log every excerpt, re-encoding, mixture and public-dataset ancestor. Search
   filenames and original video IDs across corpora. Audio hashes only find some
   copies; inspect suspected excerpts with aligned waveforms/spectrograms and
   listening before resolving `provenance_review`. Treat uncertain ancestry as
   a single source component. This can leave too few groups: collect more, do
   not split correlated windows to satisfy a quota.
2. Use two independent annotators for final audio. Mark audible event onsets and
   offsets at 0.1 s resolution, corroborated by flight logs or synchronized video
   where possible. Review disagreements together and mark unresolved spans as
   ambiguous. Review the entire recording, including negative stretches. Keep
   annotator IDs, annotation revision, review evidence and confidence. A filename
   or a visible but inaudible drone is insufficient evidence of acoustic presence.
3. Capture both positive and negative audio with each device/site and the same
   settings. Include independent stationary background sessions with fans, leaf
   blowers, lawnmowers, road traffic, speech, music and aircraft. Capture drones
   with interference in the same continuous recordings. Do not collect all
   positives on one microphone and all negatives on another.
4. Include stationary hovering, movement and changing interference. Log drone
   model and measured distance when available. For SNR experiments retain matched
   background measurements and define the band and measurement method; a mixture
   ratio is not a measured capture SNR. Retain synthetic/playback origin labels.
   Mark clipping and wind. Do not extrapolate range from a few successful clips.
5. Keep training/development sessions separate from a custodian-held final set.
   Reserve at least one unseen source, device and site when feasible. Audit
   provenance/quality before model selection without inspecting final outcomes.
   Freeze manifest/splits, targets, candidate, comparison arms, alarm policy and
   demonstration IDs before running final inference. Do not choose recordings
   based on whether the model predicts them correctly.

Provisional minimum **final** evidence in contract v1 is 100 eligible events
(each at least 3 s), 30 independent recording/session components, 10 negative
hours, two sources, two known microphones and three drone types. These are
engineering choices, not user requirements or sufficient proof by themselves.
Stratify the final report by device, site, interference, drone, and known range.
More data will be needed for subgroup bounds. Thirty cuts from one flight are
one session, not thirty independent trials. Ten negative hours with no alarms
gives a two-sided 95% Poisson upper bound around 0.37/h only under that rate
assumption; environmental diversity and cluster uncertainty still matter.

For accessible live hardware, first verify device/rate and loopback timing with
known impulses. Implement and test a causal resampler if capture is not 16 kHz;
the current file adapter's polyphase resampling is offline. Measure hardware and
OS buffer latency, dropped blocks and sustained runtime. Compare a saved capture
against replay through the same engine. Loudspeaker playback, recorded replay
and direct capture must remain separately labeled. No live-input command is
provided until this can be verified.
