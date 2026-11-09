# Frozen evaluation and artifact conventions

The JSON contract is authoritative for this experiment. Its numerical targets
are provisional. Training and tuning touch only their respective split's
waveforms; the manifest metadata is examined to check leakage and allocate
groups. Auditing the complete audio for provenance is not model evaluation.

Three-second contexts begin every 0.5 s. A decision is available only after the
end of its context; scores are never assigned to context start. Arithmetic
channel averaging and deterministic file resampling yield mono 16 kHz float
amplitudes. There is no whole-file RMS normalization. A 512-point Hann STFT with
160-sample hop feeds 48 triangular Mel bands (80–7600 Hz), power-to-dB, and
temporal mean/std. The relative alternatives subtract the current window's mean
log-Mel level. A scale-relative spectral floor limits numerical underflow;
silence below RMS 1e-5 is explicitly scored zero. Inputs shorter than 0.25 s are
insufficient, with null score and no new alarm. Silence is not an out-of-scope
decision; real low-SNR behavior is unmeasured.

Training uses full windows strictly inside one verified event or verified
negative span. It never propagates weak file labels to every window. Partial
boundary windows are excluded from supervised training. Standardization is fit
only on training features, with equal initial recording mass and then balanced
window-class mass. The logistic classifier has C=1. Optional scalar-gain
augmentation uses training audio only and caps gain before clipping. No
normalization statistics are guessed. Replay/evaluation include the **complete**
recording, boundary transitions and, when new audio remains after the last full
window, one explicitly zero-padded end-of-file window. Live streams never flush
until they actually end. Thus a short event near a file boundary can behave
differently from a never-ending stream; this convention is visible in traces.

Each raw training package is uncalibrated. `tune` scores development once and
selects from the contract's fixed threshold/confirmation grid. Hysteresis uses
off threshold = 0.75 × on threshold and two low scores to release. Successive
alarm intervals separated by at most 1 s are merged. Selection prefers feasible
**point** false-alarm rate and then recall/delay; it is not a confidence-bound
claim. Across experiment arms, the presaved plan also considers development gain
sensitivity. Only the prespecified seed-42 artifacts are candidates; other
augmentation seeds diagnose sensitivity. Tiny floating-point gain-drift
differences can break ties, so a selected arm is not evidence that augmentation
improves authentic capture or even materially improves this synthetic task.

Eligible events are at least 3 s long. Matching uses alarm **trigger** time,
with a 0.5 s collar around the event, and one alarm can match at most one event.
An alarm already active well before the event gets no free credit. Events
shorter than 3 s are counted separately and never become negative exposure.
False alarms are unmatched triggers outside the union of event collars and
ambiguity. Duplicate triggers inside positive collars remain unmatched and
reduce event precision, but do not inflate negative-time rate. Negative
exposure excludes all event collars and ambiguous spans. Negative alarm-time
fraction exposes long latched alarms; FA/h alone can hide them.

The event precision denominator is all eligible alarm events (excluding triggers
inside ambiguity); recording precision/recall uses any alarm versus any annotated
event. A recording error means any missed eligible event or false alarm. Recall,
FA/h, detected-event delay, recording confusion, source breakdowns, counts and
durations are separate fields. Delay includes context, scheduling, block waiting
and measured compute. Missed events do not enter the delay distribution; recall
must always accompany it. Real-time factor includes input checksum, decoding,
resampling, features and decisions, excludes one-time imports/model load and
report serialization. First-process invocation is tested separately. Device and
OS buffering are unknown; these are replay timings on the reported CPU.

Uncertainty resamples entire session/duplicate components 2,000 times. A
stationary-Poisson exposure interval and independent-event binomial interval are
also shown with their assumptions. Conservative upper/lower bounds prevent zero
alarms or a perfect tiny sample producing spurious certainty. None of these
models establishes unobserved source independence, and the minimum authentic
coverage checks are mandatory for a claim. A development evaluation can never
support the final claim, regardless of its scores.

Model packages contain JSON configuration/class mapping/contract, train and
development provenance, versions, code fingerprint, all threshold trials and
NPZ coefficient/mean/scale arrays. Loading checks content identity, shapes,
finiteness and SHA-256; pickle and substitute arrays are not accepted. Metadata
contains measured wall time, so regenerating a package need not reproduce the
same model ID even when predictions are identical. A source SHA identifies
uncommitted implementation contents in addition to Git HEAD.

`evaluate --split final` writes the frozen comparison plan **before** scoring
and appends a local adjacent final-use ledger. Repeated overlap is refused unless
`--allow-final-reuse` explicitly downgrades it to development evidence. Run all
prespecified baseline/candidate comparisons in one call. The ledger is a local
guard, not an access-control system: moving/deleting it, changing filenames or
looking at final results elsewhere does not restore independence. Keep it with
the dataset. If results guide further changes, declare final contamination and
collect a fresh assessment set. Replaying the frozen system for a disclosed demo
does not create new evidence.
