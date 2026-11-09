# Corpus feasibility inventory, 2026-11-09

**The acquisition gate has not passed.** This is a source/metadata audit, not a
training manifest or a new detector result. Unique usable hours, reviewed labels
and independent coverage have not yet been established for these candidates.
No detector scores were generated and no train/development/blind assignment was
made. The existing scored sources remain consumed development material.

The [second acquisition audit](../corpus-acquisition-v2/README.md) now resolves
AuDroK public file access, acquires a native pilot and adds RealMAN/DADS candidates.
The findings below preserve the first-pass scope; the current registry has eight sources.

The [candidate registry](../../data/corpus-candidates-v2.json) records the current sources,
roles, licenses, limitations and the next action for each. The
[publisher file list](../../data/candidate-files-v2.jsonl) records 202 distribution
members across four API-backed releases, with sizes, URLs and publisher checksums.
Those are distribution artifacts, including documents/archives; they are not 202
independent recordings. Full transfers and publisher checksums remain unverified.

## Findings that affect acquisition

| Candidate | Evidence inspected | Useful coverage | Limitation / next action |
| --- | --- | --- | --- |
| [Western Sydney Air 3](https://doi.org/10.26183/07nm-4p35) | README, license, directory and four alignment JSONs | Four flight recordings over three dates/two sites; about 75 flight-minutes claimed; one approximately 574-second explicit ambient tail | One drone model. Group both March flights by date/site. Resolve October background inconsistency and audit raw extents/labels. |
| [DroneNoise v3](https://doi.org/10.17866/rd.salford.22133411.v3) | Pinned metadata; all 174 WAV prefixes | Real field measurements with multiple microphone views | One campaign. 165 non-calibration files represent 59.5 file-minutes; 20 filename-based candidate event groups sum to 7.2 maximum-duration group-minutes. Neither sum is verified unique time. |
| [IDMT Berne 2022](https://zenodo.org/records/17525811) | Publisher API and file manifest | Eight flights, approximately 106 minutes claimed; static, linear and random flight, position logs | One campaign, excerpts already cut to 30 seconds; rebuild parentage. No verified negative exposure established. |
| [Polish UAV/speech](https://zenodo.org/records/15190811) | API and publisher measurement datasheet | 17 UAV units, repeated models, five date/site combinations at four locations, two recorder types | Hovering grid, not the missing moving-flight coverage. Inspect paired-device/derived files before counting hours. |
| [NEAPTIDE](https://zenodo.org/records/10512044) | API and publisher release description | Four models and several manoeuvres | Segmented, propagation-corrected measurements; establish suitability as field input. CC BY-NC 4.0, distinct from the other candidates' terms. |
| [AuDroK](https://mobilithek.info/offers/605778370199691264) | Original offer identified; [primary paper](https://www.mdpi.com/2504-446X/8/3/105) | Paper claims 40 models and 23.42 hours across original/collected material | Bulk access/terms unresolved. Audit laboratory/field/web origins and overlaps with already consumed sources before accepting any hours. |

API licenses for DroneNoise, Berne and Poland are CC BY 4.0. Sydney's publisher
LICENSE specifies CC BY-SA 4.0. File size, download statistics and repeated
microphones are never substituted for elapsed unique time. The primary papers'
aggregate amounts are publisher claims, not newly acquired quantities here.

## Specific split and label issues

Sydney's published split sends adjacent halves of one background tail to training
and test. Our session-level protocol keeps that entire capture together, along
with the other March flight. Four flight IDs therefore give three conservative
date/site groups. The README's opening table attributes approximately one minute
of background to October, while a later section says all negatives come from the
March tail. October's alignment JSON has no explicit negative interval. Keep
October negative exposure unresolved pending raw/code inspection. Flight
coordinates establish physical presence, not audible onset and offset.

DroneNoise's microphone suffixes collapse 165 non-calibration filenames to 20
candidate event keys. Some group members have different durations. The audit
uses each candidate group's maximum duration solely to expose the scale of
repeated views; that calculation does not certify unique time, complete event
identity or non-overlap. All files still belong to one known campaign.

## Evidence and reproduction

- [API snapshot facts and response hashes](source-metadata.json)
- [DroneNoise header measurements and prefix hashes](dronenoise-headers.jsonl)
- [DroneNoise aggregate calculation](dronenoise-summary.json)
- [Sydney provenance facts, document hashes and segment metadata](wsu-provenance.json)

Source metadata was retrieved on 2026-11-09. Raw responses and bounded prefixes
remain in `runs/corpus-inventory-v2`; original audio was not bulk-downloaded.
Headers were fetched with exact HTTP ranges, limited to 4096 bytes per WAV
(712704 prefix bytes total, including any samples inside those prefixes). No
waveforms were decoded, listened to or scored. Prefix hashes are not full-file
checksums. API response hashes identify the retrieved document; live counters
can change even when release members remain stable.

To repeat the header audit into a new local output directory:

```bash
curl -fL https://api.figshare.com/v2/articles/22133411/versions/3 \
  -o /tmp/dronenoise-v3.json
.venv/bin/python scripts/audit_dronenoise_headers.py \
  /tmp/dronenoise-v3.json runs/corpus-inventory-v2-repeat
```

The parser checks bounded reads, supported PCM/float headers, padded RIFF chunks
and frame extents. Tests include refusing an ignored HTTP range without reading
the full response body. Re-running this source inventory does not create fresh
independent acoustic evidence.

## Next acquisition work

First resolve AuDroK original-campaign access and source parentage, and inspect
Sydney/Berne archive or file extents and label-generation metadata. For Sydney,
select a fixed raw microphone view per flight rather than downloading video or
regenerating the approximately 2TB beamformed cache. Original channels remain
linked as simultaneous views. Resolve the label conflict before using October
negative time.

In parallel, identify substantial independently captured background timelines
and dynamic-flight campaigns with usable annotations. General environmental
class labels can support auxiliary training, but do not automatically certify
drone-absent benchmark hours. These candidates do not yet establish the proposed
100 hours, 50 sessions, dynamic-flight fraction or 30 blind negative hours.
Do not launch another small training study to compensate for that gap.
