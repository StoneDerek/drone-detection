# Source acquisition and validation, 2026-11-09

**Original AuDroK files are publicly accessible, and a native pilot has been
acquired.** A larger original-outdoor intake is now executable in Colab. The
corpus acceptance gate remains open: neither downloaded bytes nor repeated
microphone views establish the required independent, labelled capture exposure.
No detector was fitted or scored in this work.

[Open the CPU source-intake notebook](https://colab.research.google.com/github/StoneDerek/drone-detection/blob/main/notebooks/source_intake.ipynb).
Its default profile downloads **6,889,850,761 bytes (6.890 GB)** to Drive, including
two original outdoor archives. Confirm the available allocation, run all cells,
flush Drive and rerun after runtime replacement to verify completed-file reuse.
This intake uses its own pinned code folder, preserving the GPU recovery probe.

## AuDroK: access resolved; the public catalog is incomplete

The public website's JavaScript uses
`/mdp-api/mdp-msa-metadata/v2/offers/<id>`. Six offers expose metadata and file
paths under `/mdp-api/files/aux/<id>/<filename>`. Both the SQL catalog and a ZIP
returned public HTTP200 without authentication. The server ignored Range on
those probes, so large archives need full-file transfer. The
[offer snapshot](audrok-offers.json) records URLs, sizes and response hashes,
including duplicate archive names/sizes across offers.

The 3.364MB XML catalog was downloaded and its Base64 SQL body decoded as text.
[`audit_audrok_catalog.py`](../../scripts/audit_audrok_catalog.py) parses literal
rows without executing SQL. The [catalog audit](audrok-catalog-summary.json)
finds absent referenced origin IDs 51/52 and eight absent drone-type IDs. Its
outdoor-only single-drone table is empty, while later outdoor ZIP offers exist.
It is a June 2023 snapshot, not the complete current data manifest.

Its 3,276 own-campaign single-drone metadata rows sum to 10.44 **file-hours**.
Grouping their measurement-number prefixes gives 78 candidate groups of 42 rows
each; summing each group's maximum duration gives 23.27 minutes. This is a
filename diagnostic, not verified unique time. The older catalog therefore
cannot be used to equate row/file totals with field exposure or complete labels.

The acquired `MultiDrone.zip` is 200,947,757 bytes. Its complete SHA-256 was
recorded, all eight WAV member CRCs checked, and metadata decoded at native
44.1 kHz/float32. The two named measurements each have four recorder views:
20.55 file-minutes collapse to 5.14 candidate measurement-minutes. No microphone
normalization, resampling or model scoring was applied. See
[member/frame/hash evidence](audrok-native-pilot.json). This is successful source
acquisition and integrity checking, not independent confirmation of acoustic labels.

Offer rights metadata says `PUBLIC` and `NO_LICENSE`; no standard CC license is
specified. External collected sources retain their own provenance/terms. We do
not relabel the collection as uniformly CC0, CC BY or MIT.

## Sydney: explicit interval bounds must be applied in our own pipeline

Four fixed-recorder headers were read through bounded HTTP206 requests. Each
has six channels at 48 kHz in PCM32. Their complete frames total 93.64 raw minutes,
including material outside supported flight/ambient labels. Both selected March
files contain eight additional bytes after their last complete interleaved frame.
The default strict header parser rejects this condition; diagnostic mode reports
it explicitly. The known ambient bounds end before that incomplete final frame.
No source bytes were silently repaired. See [header evidence](wsu-fixed-view-headers.json).

The released generator's `load_segment_config` preserves start time but does not
read `skip_seconds_end`. Its negative loop continues to EOF. Consequently, as
released, processing `no_drone_1` would also process `no_drone_2`. Static inspection
found no end-bound references; [source identity](wsu-generator-inspection.json)
records the exact script hash. We did not execute that external generator or
reproduce the paper's NPZ/training results, so this finding does not establish
which code or split produced the published results.

For future ingestion, enforce these publisher intervals explicitly as half-open
native-frame ranges; do not inherit the released generator's slicing behavior:

| Interval | Start/end seconds | Start/end frames at 48 kHz | Duration |
| --- | --- | --- | --- |
| `no_drone_1` |1290–1576.9 |61,920,000–75,691,200 |286.9s |
| `no_drone_2` |1577–1864 |75,696,000–89,472,000 |287.0s |

Our split keeps the whole March capture together regardless of the publisher's
train/test labels. The October negative-time discrepancy remains unresolved;
its config supplies no explicit negative interval. Flight telemetry continues
to indicate physical presence, not reviewed audible onset/offset.

## Berne: directory facts differ from the broad duration claim

The ZIP directory contains 163 real WAV members after excluding macOS resource
forks. Filename roots suggest eight flights, all dated 3 August 2022, with multiple
source-part suffixes. One inspected WAV is 30 s, eight-channel48 kHz/PCM16. Member
sizes are consistent with a nominal 81.5 file-minutes, rather than the publisher's
approximately 106-minute description. This remains provisional until every header
and timing/coordinate mapping is verified. Do not concatenate unknown part
suffixes or claim 81.5 accepted unique minutes.

The parallel all-header probe received HTTP429. It was stopped; remaining
header/CSV verification is incomplete. Reuse the cached central directory and
serialized access, or download the 1.515 GB archive once in the cloud. The
[directory evidence and limitation](berne-directory-summary.json) and
[member list](berne-members.json) preserve what was established. No whole-archive
checksum, reconstructed timeline or acoustic annotation is claimed here.

## RealMAN: substantial background candidate and selective-transfer proof

The [official project](https://github.com/Audio-WestlakeU/RealMAN) releases raw
background separately from loudspeaker speech recordings. Its pinned scene
metadata totals 8665 rounded minutes across 31 noise scenes, close to the paper's
144.5 h. Outdoor exposure is 28.45 h; semi-outdoor 16.65 h; transportation 18.78 h;
indoor 80.53 h. Scene counts do not establish capture-day/device independence.

The 51 multichannel noise archives total478.6GB. A sampled archive stores FLAC
members without additional RAR compression, enabling direct byte-range transfer.
One 1,984,071-byte CH0 member was fetched with HTTP206, checked against its RAR CRC,
hashed, and decoded as 60 s of native48 kHz/PCM16. The other simultaneous channels
were not transferred. This proves selective extraction for that member, not for
every archive layout. [Noise inventory](realman-noise-inventory.json) and
[member proof](realman-member-proof.json) pin the release and measurement.

These are useful auxiliary background candidates. They do not supply independently
reviewed drone-absent benchmark hours, and a negative-only source does not solve
the need for both classes under comparable capture conditions.

The DADS aggregation was also checked. Its card claims 27.10 positive hours, but
mixes ground, onboard and fault sources and repeats several candidate/consumed
collections. Its visible schema has audio and a binary label; inspect original
audio paths for recoverable parentage before accepting a subset. Mirrors and
extra subsecond rows do not meet the missing independent-field requirement.

## Executable intake and next gate

[`source-intake-v2.json`](../../data/source-intake-v2.json) defines two profiles:

| Profile | Bytes | Purpose |
| --- | ---: | --- |
| `pilot` |206,296,045 |Catalog, verified dual-drone ZIP and one verified native background member |
| `outdoor` |6,889,850,761 |Pilot plus the two original outdoor WAVE ZIPs |

[`fetch_source_intake.py`](../../scripts/fetch_source_intake.py) checks declared
budgets and transfer lengths, verifies pinned pilot hashes, records first hashes
for the large archives, verifies storage read-back and publishes completion
receipts. Reruns rehash completed files. An interruption during an AuDroK archive
requires restarting that archive because the server ignores ranges. Incomplete
files are not accepted as complete. Mounted read-back is not independent proof
of cloud sync; verify again after actual runtime replacement.

Live [full-file catalog and exact-range member checks](downloader-live-check.json)
passed with the new downloader, including completed-file reuse. Local pilot
recovery also verified the existing ZIP. Corrupted payload/receipt, truncated/
oversized body, ignored-Range and pending-receipt recovery paths have automated tests. **The 6.684 GB pair of outdoor archives and
actual Drive intake have not been run here.** The current executor lacks bulk
space and authenticated Drive access; the notebook is the execution handoff.

Next: acquire the original outdoor pair; audit native precision, repeated views,
flight state and current source labels; expand the fixed-channel background
inventory; and resolve remaining independent positive/negative coverage and
annotation needs before creating the study's frozen manifest. Do not replace
that gap with another small model-training cycle.

Source credits: AuDroK—H2 Think/Fraunhofer IVI; Berne—Ximei Yang, Jörg Bitzer
and Fraunhofer IDMT; Sydney—Belman Jahir Rodríguez, Sergio F. Chevtchenko and
Saeed Afshar/Western Sydney University; RealMAN—Bing Yang et al., Westlake
University and AISHELL. Original source terms are retained in the inventories.
