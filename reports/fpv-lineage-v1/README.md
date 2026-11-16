# FPV waveform and lineage qualification

The complete FPV archive from [Zenodo 21935566](https://zenodo.org/records/21935566)
has been acquired, checked against its publisher MD5 and audited without classifier
inference. **Its prebuilt train/validation/test splits are unsuitable for measuring
transfer to an unfamiliar capture.** All three observed capture-folder groups occur
in multiple publisher partitions. They represent a small number of capture groups,
regardless of the number of exported clips.

The archive remains an **unscored candidate**, outside the running BEATs experiment.
This is a data qualification result, not evidence of detector accuracy. The
[aggregate evidence](audit.json) pins the code, inputs and private ledgers.

## Complete inventory and decode results

| Material | Members | What the count means |
|---|---:|---|
| Longer publisher-labelled intervals | 249 | 128 drone and 121 background paths; labels remain weak |
| Prebuilt split exports | 19,012 | Derivatives, including 13,328 overlapping augmented-training windows |
| Other/raw audio | 25 | Supplied for one of the three capture-folder groups |
| Total outer audio decoded | **19,286** | Every identified WAV/M4A member; zero remaining decode failures |

The one nested ZIP lists six JPEG members and no audio members. Its images were
not used. The complete 2,152,999,558-byte FPV ZIP matches publisher MD5
`b06064de415a1d8fdf16b3e508ba96a0`; its observed SHA-256 is
`6c36545a296881d50331af0860d312529d864571c5b9e4158cd522b2b7f35f99`.
This whole-file verification applies to **Fpv.zip only**, not the release's electric
or general-UAV ZIPs.

Four nominal `.wav` labelled intervals actually contain valid AAC/M4A. An initial
libsndfile-only pass failed on them; checking the container and decoding with
FFmpeg recovered all four. They are not established corrupt recordings. The fixed
decoder preserves native rate/channels and emits PCM16 before float32 comparison.
Other WAVs use native-rate/channel libsndfile float32 decoding. Raw-file hashes are
retained separately; there is no resampling, channel averaging or gain fitting in
the identity checks. The native-format counts and FFmpeg version are in the JSON.

## Verified relationships and unresolved material

| Direct export-to-declared-original outcome | Exports |
|---|---:|
| Unique exact decoded-sample subsequence | **7,504** |
| No included, uniquely identified parent under the declared alias | **11,476** |
| Available declared parent, but no exact native-sample match | **32** |
| Total accounted | **19,012** |

The missing-parent outcome does not prove the same audio is absent under another
name. The no-match outcome does not prove the export is unrelated: re-encoding or
other processing can break exact equality. Neither outcome is silently promoted
to a verified relationship. Filename copy suffixes are preserved; one such export
was resolved by its actual waveform rather than by stripping its suffix.

There are 165 groups of identical decoded export waveforms. None spans publisher
partitions or class labels. For the 7,504 verified direct matches, 236 provided
parent files have matched exports, and 193 of those parents occur in multiple
partitions. **No cross-partition sample overlap was measured in that verified
subset.** The remaining exports are unassessed for sample overlap. Sharing a source
file or capture is not, by itself, proof that identical samples cross partitions;
conversely, zero measured overlap does not establish capture independence.

The filename-declared augmented-training windows sum to 13,242.593 seconds, with
8,424.597 seconds of union on their respective declared alias timelines. The
36.38% overlap illustrates why window counts cannot be treated as independent
recordings. These sums are not unique physical capture time: views may be parallel
and the declared timelines are not all waveform-verified. Ninety exports are
shorter than 250 ms and would fail the existing study's duration rule.

## Labels, raw sources and capture groups

All 249 labelled intervals were checked against all 12 released FPV annotation
documents. Filename/export-numbering hypotheses and native durations provide
151 unique duration-consistent candidates, one ambiguous candidate and 97 intervals
without a matching released annotation. Nine intervals intersect opposite-class
annotation overlaps, including small boundary-rounding differences. Candidate
numbering hypotheses and original labels are retained. A duration-consistent
candidate is not a verified continuous-source timestamp or independently reviewed
audibility label.

Comparing the intervals with the 25 available raw-view files established 12 unique
exact matches; those pairs are also byte-identical. Another 39 intervals have no
exact match in an available matching capture/device view, and 198 have no supplied
matching raw view. This search does not invent continuous sources for other
captures or search every unrelated filename/view.

The conservative capture-folder groups contain 51, 93 and 105 labelled intervals,
respectively. Their 2, 2 and 8 view folders are not evidence of independent recording
campaigns. The JSON publishes anonymous group IDs and class counts. Do not divide
simultaneous views or derivatives between model development and confirmation.

## Prior project exposure check

All 19,286 member byte hashes were compared with recorded stored-file/native-member
SHA-256s in the current 30,238-crop manifest and five historical manifests. The six
manifests contain 34,596 rows in total, with overlap between manifests. There were
zero recorded whole-file byte matches. The current manifest provides native-member
hashes for 28,562 rows; the other manifests primarily supply stored-file identities.
Coverage and manifest hashes are explicit in the JSON.

This is a limited exact-identity screen. It does not rule out shifted excerpts,
different crops, lossy copies, gain changes, rewrapped audio or model-pretraining
overlap. Project-wide recording freshness therefore remains unresolved.

## Disposition

Preserve the 249 original labelled intervals as weak-label candidates, grouped by
capture. Keep unresolved derived exports out of any qualified cohort. Do not adopt
the publisher split or count 19,012 exports as new independent evidence. No data
from this archive has been added to training or scored by a detector.

The remaining qualification work is source/capture provenance, broader excerpt and
waveform-overlap checks against prior project material, continuous-source/annotation
alignment where possible, and label/audibility review. Capture diversity must remain
explicit even if every interval becomes usable. This release alone cannot establish
broad field generalization. The anonymous publisher creator entry and CC BY 4.0
terms are preserved; authorship is not independently established.

## Reproduction and retained evidence

[REPRODUCE.md](REPRODUCE.md) describes acquisition from the pinned public archive,
input recovery, all waveform/lineage stages and aggregate publication. Recovering
inputs afresh from the complete ZIP produced the same directory entries, all 12
FPV annotation records and all 19,012 declared-lineage rows as the initial cached
range audit. A partial catalogue check is not substituted for the complete audit.

Complete directories, original annotation text and per-member ledgers remain in
ignored local evidence storage. Public output contains counts, anonymous group
identifiers and hashes. No audio, images, absolute audio paths or complete member
filename inventories are published. Software checks passed: **198 tests**, nine
optional skips, and Ruff. These checks are not drone performance measurements.

The running BEATs recipe, source folds, numerical trainer and cumulative 44-hour
budget remain unchanged. Its partial-epoch diagnostics remain separate from this
source qualification.
