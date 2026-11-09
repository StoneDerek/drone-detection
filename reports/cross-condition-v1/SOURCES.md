# Source identities, labels and limits

The model packages were fixed before each cohort's first scores. Full model
identities, original-file identities, source revisions and execution ledgers are
retained beside the results. All scored cohorts are now consumed for subsequent
development. No new listening annotations have been assigned.

## Kenya Forest Sounds

- **Primary record:** https://doi.org/10.5281/zenodo.21443074
- **Creators:** Allan Vikiru, Daniel Simiyu, Sandra Kahoro, Zainabu Muti, Solomon
  Kamau, Anthony Macharia, Delphine Clara Zemp and Henry Muchiri.
- **License:** CC BY 4.0.
- **Repository:** https://github.com/SCES-Makerspace-Lab/Kenya-Forest-Sounds-Dataset
- **Pinned revision:** `d451f19df9c6aaf4acf1758b247a6209a0d8406e`.

The full exact revision and eight Git-blob identities are in `source-freeze.json`.
The five selected WAVs and the README, workbook and datasheet passed Git blob SHA-1
and byte-length checks. Original publisher metadata is copied into `source-records/`.
The repository was inspected using the GitHub connector; binary downloads used
HTTPS from that pinned revision. The workbook parser checks the human-annotated
tab's Filename/Label headers and column C; a secondary Drone comment in another
column is not promoted. Exactly the five primary Drone rows were selected.

These are field excerpts with human dominant-sound clip labels, not reviewed
audible onsets/offsets. All five filenames share the apparent original ZOOM0071.
Other primary classes can coexist with a drone, so they are not certified
negative intervals. Whole-file, canonical decoded and gain-normalized hashes
found no match against the old and external-v3 corpora; these checks do not
exclude every shifted or lossy derivative. The inference plan and first-use
ledger were written before scores. This is one new positive capture condition.

## One-channel controlled-capture source

- **Primary record:** https://doi.org/10.5281/zenodo.23613504
- **Title:** A 16-Channel Acoustic Dataset for Passive Ground-Based Monitoring
  of Small Drones.
- **Creators:** Huilong Wang, Peng Chen, Zhimin Chen and Yuye Liu.
- **License:** CC BY 4.0.
- **Release:** 1.1.0, original main archive plus 2026-11-09 maintenance record.

The maintenance documentation says the main 273 recording JSONs and 4,368 WAVs
are unchanged; the separate sweep calibration is excluded. Only the original
main archive's metadata, recording JSONs and channel-0 WAV members are used.
Author benchmark code, weights, folds and instructions are not executed.

All 200 experimental hovering-drone recordings, 28 field backgrounds and 45
controlled loudspeaker replays are selected before inference. No selection is
made by distance, scene, score or apparent difficulty. One microphone channel
is fixed across the entire source. Source FSD50K identifiers remain in replay
JSON provenance, and those replays are reported separately from field audio.

Exact HTTPS ranges check response status and Content-Range. ZIP-member CRCs,
byte counts and publisher manifest SHA-256 values are checked. The file index
and manifest must agree. Verified WAVs are converted to lossless FLAC and decoded
back to identical signed 16-bit PCM before the receipt is written. Original WAV
hashes, retained FLAC hashes, PCM hashes, canonical fingerprints, original JSON
bytes and the publisher manifest remain available. Partial fetching does not
verify the whole archive's MD5; every report states this limit.

The publisher gives full-recording positive extents with
`label_source: experimental_design`. Automated release QC is not a human audible
boundary review. These labels support a recording-level experimental contrast,
with no claimed event recall, false alarms/hour or verified detection radius.
The 215 publisher session IDs include per-recording placeholders. The session
table explicitly marks historical dates unavailable for all 200 positives and
12 backgrounds, despite date-like IDs and dates in other metadata. A separate
metadata-only amendment, frozen before inference, pools those 212 recordings
together and uses the two explicit October acquisition dates for the other 61.
These three conservative metadata components are not three independent sites
or a statistical independence estimate. April–July batch names remain publisher
descriptive fields, rather than established independent capture dates. The
classes are not balanced within those batch labels. One site, array and drone
family also limit generalization even when recording counts are large.

The source was discovered after the Kenya and consumed-FPV outcomes. All twelve
earlier models and thresholds are carried forward; none was selected or tuned
from this source's audio. This timing is recorded in its separate plan.

## Earlier consumed sources

The 615 old train/development recordings retain their publisher labels and
conservative provenance groups. All 37 FPV recordings were scored earlier in
external-v3; their original Git blob identities and attribution remain in
`../external-v3/`. No original final-assessment recordings are newly relabelled or
counted as fresh here. The twelve classifiers were fitted on ITU only, but the
project had already inspected these transfer sources. An explanation using
their retained features is diagnostic evidence, not another independent test.
