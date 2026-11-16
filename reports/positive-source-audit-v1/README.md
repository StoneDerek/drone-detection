# Public positive-source audit

The highest-value next data milestone is to recover original capture/annotation
relationships in the **FPV portion of a newly identified public field release**.
It has publisher drone/background intervals under the same recording conditions.
That addresses a gap the current baseline mixture's 976 positive crops from four
capture groups cannot settle. This audit does not add data to the running study,
score a new model or claim that more positive data will necessarily help.

## Corrected project exposure

The previous BEATs reservation incorrectly implied ITU ARIS was untouched in the
project. [Earlier results](../external-v3/RESULTS.md) already evaluated two CNNs,
YAMNet and an SVM on it. [Within-session learning](../human-label-task-v1/RESULTS.md)
and [transfer diagnostics](../transfer-diagnosis-v1/RESULTS.md) also used it.
Its v1.1 publisher description explicitly says audio and splits are identical to
v1.0. A documentation update cannot restore test freshness.

The corrected [reservation record](../beats-adaptation-v1/future-source-reservation.json)
now separates this study's activity from historical project exposure. FPV/Saarmets
and Nanjing are also already consumed project collections, despite being outside
the current BEATs training manifest. They remain useful for disclosed development
or future training; they cannot supply untouched confirmation by being renamed.
[The exposure ledger](exposure-ledger.json) pins the historical evidence.

## Candidate findings

| Source | Verified metadata/catalogue findings | Appropriate next role |
|---|---|---|
| [Acoustic UAV Field Recordings](https://zenodo.org/records/21935566), CC BY 4.0 | Three public ZIPs total 4,175,800,607 bytes. All three central directories and all 21 TXT annotations were inspected. The directories contain 32,465 WAV members, including raw/longer intervals, channel views and prebuilt split exports. | Prioritize the FPV recordings for provenance/label qualification. Reserve the release from model scoring while that work proceeds. Electric/general UAV category meanings still need verification for this project's target. |
| [CiO-Drone-Acoustic](https://zenodo.org/records/23170386), CC0 | Publisher-verified README, catalogue and channel audit. Thirteen indoor takes, 3,165.7017 seconds of raw reference audio, with sidecar dates on two days. Reference and array views describe the same takes. | Candidate auxiliary training for aircraft/hardware variety. Use native mono reference audio rather than an averaged array view. It is not an outdoor confirmation set; sidecar dates alone do not prove independent sessions. |
| FSD50K metadata | All 51,197 metadata entries were scanned. 197 contain the word "drone". The only match to a stricter airframe-keyword pattern describes a **PC fan**, with rotor/drone tags. | No new positive training pool was established. A tag or filename cannot automatically become a drone label. No new audio was fetched for this check. |

The field release's FPV ZIP contains 249 non-split, labelled interval WAVs across
three candidate capture-folder keys: 128 under drone paths and 121 under
background paths. It also contains 19,012 publisher split-export WAVs. These are
different levels of the same collection, not 19,261 independent recordings.
Phone folders may be simultaneous views; until alignment is verified, group
them conservatively by capture. Device names do not create new capture trials.
The electric ZIP similarly contains 72 non-split labelled WAVs
and 12,940 split exports. The general UAV ZIP's labelled paths are positive-only.

The 21 annotation documents contain 343 parseable intervals. Seven documents
have opposite-class overlap at some boundary; only one has overlap greater than
10 ms, with a maximum of about 0.720 s. Preserve those ambiguous regions rather
than silently trimming them into clean truth. Numeric label suffixes and the
publisher spelling `eletric_uav1` are explicitly mapped while raw labels and
document hashes are retained. Text validity is not waveform alignment or
independent audibility review.

CiO's catalogue also notes that the aircraft specification for the Neo folder
describes Neo rather than Neo 2. Binary drone training may still be possible,
but aircraft identity should not be asserted beyond the available evidence.
Its array has channel-status issues documented by the publisher; the mono
reference is the appropriate first input candidate.

## Concrete next milestone

1. **Build the complete parent manifest for FPV.** Map every labelled interval,
   derivative and publisher split export to its source phone recording and
   capture folder. Link simultaneous views and keep an entire capture group in
   one partition. Do not adopt the publisher's train/test export merely because
   it is already split. Size/CRC coincidences are duplicate candidates, not
   decoded-audio identity proof.
2. **Verify the eligible audio and interval mapping.** Preserve native source
   checksums, decode identities and source terms. Select a physical microphone
   consistently across positive/background intervals. Check labels against
   actual available durations and quarantine conflicts, gaps and uncertain
   boundaries. Do not reconstruct missing continuous audio from assumptions.
3. **Check independence from project material.** Compare original/decoded
   identities with existing sources and investigate likely excerpt/duplicate
   relationships. A repository text search found no previous use of these two
   candidate releases, but that does not prove recording independence or rule
   out mirrors/pretraining exposure.
4. **Freeze a source-wide assessment only after qualification.** The existing
   model recipe and model selection remain fixed independently of this source.
   Use the complete eligible cohort with explicit group counts and publisher
   label limits. Keep classification results separate from continuous alarm
   claims. If the available material does not support the intended assessment,
   retain it as a development/training candidate with that narrower role.

This ordering preserves a possible confirmation source while the current
full-backbone comparison runs. CiO is secondary auxiliary training material;
adding its indoor takes now would change the experiment before its existing
control comparison is complete. There is no new hour quota, sample-count target,
custom architecture or proprietary collection requirement.

## Evidence and limits

[metadata-audit.json](metadata-audit.json) contains complete aggregate measurements
and hashes for the cached inputs. [candidate-sources.json](candidate-sources.json)
pins publisher file sizes, MD5s and retrieval URLs. Whole CiO metadata files were
MD5 checked; the field ZIPs were accessed with exact HTTP byte ranges, and all
extracted TXT members passed ZIP CRC checks. **Whole field-archive MD5s were not
verified.** Catalogue/annotation range blocks may include incidental compressed
audio bytes; no audio member was extracted, decoded or scored.

Full directories, annotation text and observed publisher metadata remain in
ignored `work/positive-source-discovery-v1`. Reproduce the aggregate analysis with:

```bash
python scripts/summarize_positive_source_audit.py \
  --input work/positive-source-discovery-v1 \
  --output reports/positive-source-audit-v1/metadata-audit.json
```

The source search was a bounded primary-repository search, not an exhaustive
claim about all public data. The running 30,238-crop study, data order,
hyperparameters and cumulative 44-hour budget are unchanged.
