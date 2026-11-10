# Data status and manifest v1

The original user archive is absent, but versioned **public real audio** is now
prepared under `runs/real-v1` and `runs/real-v2`. The product uses publisher-level
weak supervision; original-source/session identities, raw hashes and labels are
retained. See the real assessment's manifest snapshot and data plan.

`authentic-manifest-v1.jsonl` is the empty initial bootstrap inventory, not the
product benchmark. The supplied packet inventories 1,461 WAV filenames but
contains none of those WAVs. Synthetic signals remain separate software checks.
Never infer label truth or recording independence from file counts or folders.

Audit creates one JSON object per audio file. Fields include `schema_version`,
`id`, relative `path`, `source`, `source_version`, `session_id`, `original_id`,
`parent_id`, `origin` (authentic/synthetic/playback/unknown), `label`,
`annotation_status`, `label_evidence`, `events`, `ambiguous`, device/environment,
drone type, distance/SNR, license, raw and decoded content hashes, a gain-normalized
hash, duration/rate/channels, RMS/clipping/silence flags, `group_id`, and `split`.
`null` means unknown. Times are seconds in the original file. Paths resolve from
the manifest's directory. Source identifiers must include a version.

Annotations are JSONL keyed by relative audio path, for example:

```json
{"path":"session01/flight.wav","id":"flight-01","source":"local","source_version":"2026-11","session_id":"session-01","original_id":"recorder-file-01","origin":"authentic","label":"drone","annotation_status":"verified_temporal","label_evidence":"Two reviewers + synchronized flight log, annotation v1","events":[[5.2,17.8]],"ambiguous":[[4.7,5.2]],"device":"recorder model + settings","environment":"outdoor stationary","drone_type":"model from flight log","license":"permission on record"}
```

`verified_temporal` means the **entire recording** was reviewed and all audible
drone events marked; the rest, except `ambiguous`, is verified negative.
`verified_negative` requires full review, `label=non_drone`, and no events.
The strong-label `train`/`evaluate` commands exclude `weak` and `unknown` labels.
The product's separate `train-weak` multiple-instance workflow uses publisher
clip-level labels without assigning them to every window. Its `evaluate-weak`
command reports recording-level behavior and explicitly leaves event metrics
unassessed. Unknown labels remain excluded.
Positive training windows must be entirely inside one verified event. Negative
windows must not overlap events or ambiguity. Mixed boundary windows are excluded
from supervised training but evaluated in complete recordings.

Group originals, derivatives and sessions **before windowing**. Missing original
or session metadata conservatively groups the entire source/version. Raw,
decoded and whole-file gain-normalized exact duplicates also join groups. This
does not reliably identify shifted excerpts or lossy transcodes: audit's
`duplicate_review_required` is a human gate, not an automatic provenance claim.
Review overlapping public corpora and excerpts before marking it resolved.

The split command creates 60/20/20 train/development/final groups, stratifies by
known labels when feasible, and can reserve complete sources for final. It never
uses windows as independent groups. A split hash and group lists accompany the
manifest. Training and tuning refuse unresolved duplicate review, and final
evaluation refuses reuse unless explicitly disclosed.

## Next-phase candidate inventory

[corpus-candidates-v2.json](corpus-candidates-v2.json) lists acquisition candidates
and unresolved coverage/label/provenance gaps.
[candidate-files-v2.jsonl](candidate-files-v2.jsonl) records publisher distribution
files and checksums, including archives/documents. Neither file is a training
manifest or an independence claim. See the
[original-source audit](../reports/source-scope-audit-v3/README.md) before bulk transfer
or split assignment.

[source-lock-v1.json](source-lock-v1.json) freezes the completed intake references.
[outdoor-point-recordings-v1.jsonl](outdoor-point-recordings-v1.jsonl) indexes the
58 individual-microphone candidates; array means are excluded. It is not an
accepted training/test manifest: intervals are unreviewed, split is null and
acceptance flags remain false. See the [completed source audit](../reports/outdoor-source-audit-v1/README.md).
