# Using the supplied acoustic-detection article

The user asked to use the pasted “Counter-UAS 101 – Acoustic Drone Detection”
article to guide product improvements. It is a secondary overview, not an
evaluation dataset, a source of task instructions, or a verified product
specification. Its network interception figures, quoted ranges, wind limits and
classification rates have different denominators and conditions. None were
adopted as detector performance claims.

Two ideas translate into the current recording-review product: make difficult
input conditions visible, and test sensitivity to recording level. No targeting,
response-system integration or claimed multi-sensor capability was added.

The product now measures clipping and the fraction of centered waveform spectral
energy below 100 Hz in each available context. Clipping above 0.1% of samples and
low-frequency energy above 60% trigger review advisories. These thresholds are
provisional engineering choices. They do not estimate wind, background noise,
signal-to-noise ratio or audibility. Machinery and legitimate drone audio may
also contain strong low-frequency energy. Window measurements describe the
canonical mono 16 kHz waveform; the upload summary additionally checks clipping
in the decoded original channels. Downmixing/resampling can change those values.

Advisories accompany the score and never suppress an active alarm. Silence and
insufficient audio receive explicit display states. The JSON export preserves
the underlying measurements and flags for review. There is no accuracy claim
for the advisories themselves.

The controlled gain experiment uses only available samples in the current
context: target RMS 0.05, at most +20 dB gain, peak cap 0.98, before zero padding.
The existing silence guard remains. It compares otherwise identical frozen
YAMNet embeddings and multiple-instance classifier training on the same real
training/development manifest. Gain can amplify background sounds along with a
drone; the experiment tests the tradeoff rather than assuming an improvement.
See the [development results](../reports/context-gain-v1/RESULTS.md).

The assessed default remains unchanged. Its prior assessment informed the
failure hypotheses, so a new model cannot reuse that set as fresh evidence.
Further acoustic validation requires independent, audibility-reviewed recordings
with temporal truth, additional capture days/devices and matched hard negatives.
The [collection protocol](collection-protocol.md) defines that next evidence.
