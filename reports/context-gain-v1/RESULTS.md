# Context gain experiment: development only

The bounded-gain candidate detected **21/22 publisher-positive development
files**, compared with **20/22** for the assessed default. Both selected operating
points alarmed on **9/188 negative files**. This is a one-file development gain;
it does not establish improved field reliability. The assessed default remains
unchanged.

## Controlled comparison

Both arms used the same 405 real training recordings and 210 development
recordings, frozen official YAMNet encoder, 3-second context, 0.5-second hop,
multiple-instance classifier recipe, C=1, seed 42, threshold grid and
`sensitivity-v2` selection policy. Training record IDs, source/group identities,
audio hashes and sampled-window counts matched exactly. Only the available
context's amplitude treatment changed before encoding; the classifier head was
retrained and each arm's decision selected on development data.

The candidate targets RMS 0.05 with at most +20 dB gain and peak cap 0.98,
computed before padding. The existing silence guard is preserved. These are
provisional engineering choices, not values validated by the supplied article.

| Model and decision | Positive files detected | Negative files alarmed | Recall | Negative-file FPR |
| --- | ---: | ---: | ---: | ---: |
| Default; selected on=0.5, off=0.375, 2 on-hits | 20/22 | 9/188 | 90.91% | 4.79% |
| Gain candidate; selected on=0.8, off=0.6, 1 on-hit | 21/22 | 9/188 | 95.45% | 4.79% |
| Gain candidate at the default decision | 20/22 | 7/188 | 90.91% | 3.72% |
| Default at the gain candidate's decision | 20/22 | 12/188 | 90.91% | 6.38% |

All decisions retain two off-hits. The extra positive file at the selected
operating point reflects the complete feature/training/calibration recipe;
normalization is not a threshold-compatible drop-in. The fixed-decision rows
are retained from the same development grid, not additional trials on unseen
audio. [All metrics](results.json) and [fixed-decision values](fixed-decision-comparison.json)
include confusion counts and the uncalibrated score ranking measures.

## Evidence scope

The manifest contains 2,356 real publisher-labeled recordings, including 405
training, 210 development and the previously assessed 1,741-file set. This
experiment used training and development only. It did not rescore the prior
assessment as a fresh test. That assessment already informed failure hypotheses
and is consumed for future model optimization. Recordings, overlapping windows
and correlated capture groups do not establish independent field trials.
The 22 positive development files cover just four provenance groups: 11 Batear
clips in one capture group and 11 D3Set clips in three groups. This further
limits what the one-file gain can establish.

No independent audible-event boundaries, acoustic SNR or wind measurements were
added. Event recall, false alarm events per true negative hour, delay and range
remain unassessed. The article's network interception figures and quoted ranges
are not classifier specifications.

The same development traces contained 12,710 context windows: 2,806 flagged for
strong low-frequency energy and 6 for clipping. These counts describe how often
the provisional quality advisories appeared. They do not measure true wind,
advisory accuracy, coverage or an improvement in detection. The flags do not
change scores or suppress alarms.

Training took 140.16 seconds and development scoring 573.25 seconds in this run.
Other product checks ran concurrently, so these elapsed times are not a fair
speed comparison with the earlier default run. The separate
[warm profile](warm-profile.json) records same-process timings and its exclusions.

## Artifacts and use

The complete [experimental model](../../artifacts/context-gain-v1/README.md)
contains weights, train-only normalization, the frozen encoder, all development
threshold trials, labels/provenance references and explicit **DEVELOPMENT-ONLY**
metadata. It is compatible with product 0.2.0. The default startup script still
loads the independently assessed version-2 package.

The manifest SHA-256 is
`371ab29f57e857d0ac32d2f5d346b7e7518852aa10669a29321ae0e68b9950ff`.
The default model is
`b9c7df3a56936a161dbf7e70fb02b6979666bfe8043d131a4e38496da13c261f`.
The experimental model is
`464e329b3d2253f9c928eb721fc0a001006b9fb3972312f65ad773bbb49ea35d`.
The [control verification](control-verification.json) ties the data, encoder,
source snapshots and model identities together.

The next useful validation is a new independent, audibility-reviewed collection
across more days/devices with matched hard negatives. The current result supports
carrying the candidate forward to that comparison, rather than promoting it to
the default or claiming a reliability target has been met.
