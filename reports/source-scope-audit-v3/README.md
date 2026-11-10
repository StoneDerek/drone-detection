# Original-source coverage audit, 2026-11-10

**The earlier 100-hour target was an unvalidated planning estimate, not a
standard or a demonstrated data requirement.** It and the related diversity
counts are withdrawn as readiness gates in the [research plan](../../docs/research-plan-v2.md).
Training quantity will be estimated from grouped learning curves after broad
intake. Evaluation exposure has a separate precision and coverage requirement.

The small completed AuDroK outdoor subset does not establish a shortage of
suitable public data. This pass returns to **all ten original sources identified
by the user**, alongside the previously identified candidates. DADS membership
does not establish quality, independent capture time, new-test status or common
redistribution terms. No audio was scored, no labels invented and no study split
assigned. This is a metadata audit; most original audio still needs acquisition
and content inspection before a frozen study manifest exists.

## Original sources and provisional roles

| Original source | Evidence and scale | Appropriate candidate role / unresolved work |
| --- | --- | --- |
| [Al-Emadi DroneAudioDataset](https://github.com/saraalemadi/DroneAudioDataset) | Original README describes indoor propeller recordings with artificial noise augmentation. Complete Git tree has 23,408 WAV paths but 10,625 distinct WAV blob IDs; binary and multiclass layouts have identical sets of blob IDs. | Auxiliary indoor/augmentation acoustics. Recover positive raw-capture lineage; augmented files are not independent field time. Unknown filenames map to all 2,000 ESC-50 originals, in both layouts; this is a filename relationship, not a new payload-level acoustic comparison. Current custom restricted terms and third-party licenses remain separate. |
| [SPCup19 egonoise](https://dregon.inria.fr/datasets/the-spcup19-egonoise-dataset/) | Publisher lists ten team packages, multiple drone types and 1/4/8/16 audio channels, explicitly as drone-embedded recordings. Original page and documentation links are accessible. | Diverse onboard acoustics, potentially useful auxiliary training. Audit each team's documentation, raw extents and sensor placement before assigning more specific roles. Hours are unresolved; neither ten teams nor extra channels establish independent elapsed time. |
| [DREGON](https://dregon.inria.fr/datasets/dregon/) | Eight-microphone drone-mounted array; source-bearing room flights, noise-only flights, individual motor recordings and clean loudspeaker recordings, with position/motor/IMU logs. | Useful mechanism/onboard data with different geometry from ground listening. Keep clean, motor-only and source-bearing recordings distinct. Raw duration and package lineage remain to be inventoried. Publisher terms specify personal, educational and academic use. |
| [DroneNoise v3](https://doi.org/10.17866/rd.salford.22133411.v3) | Existing all-header audit: 165 non-calibration WAVs, 59.5 file-minutes from one Edzell campaign. Filename grouping gives 20 candidate event keys; their maximum durations sum to 7.17 minutes, without proving unique timelines. | Real ground-microphone overflight candidate. Recover event/microphone alignment and labels. Do not multiply exposure by simultaneous microphones or certify the grouping diagnostic as unique time. CC BY 4.0. |
| [AuDroK](https://mobilithek.info/offers/605778370199691264) | Completed original-outdoor audit: 58 point files, 2.9407 file-hours, one campaign; 174 other files are array means. Other original-lab and external-source packages remain outside that completed audio audit. | Expand to the remaining packages and original parent sources. Distinguish lab/field/web origins, XML/WAV duplicates, channels and campaigns. Public access metadata says NO_LICENSE; it is not a blanket CC grant. The 2.94-hour subset is not the collection's total. |
| [Yi/Choi/Lee fault data](https://zenodo.org/records/7779574) | [Primary paper](https://arxiv.org/abs/2304.11708): three body-mic drone models, tethered in a chamber; 0.5-second segments mixed with noise from five campus places at 10–15 dB SNR. 54,000 mixtures/model, two microphone views; three TARs total 7,622,850,560 bytes. | Auxiliary onboard/fault acoustics, not a real ground-field benchmark. Filename indices may recover base drone/background relationships; inspect these before using the publisher's randomly shuffled mixture split. Unique raw capture time is unresolved. CC BY 4.0. |
| [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) | Full CSV: 8,732 excerpts, 1,297 Freesound parent IDs, 8.7512 summed excerpt-hours but **3.9623 hours of represented source-interval union**. Native sample formats vary. | Substantial background/confuser supervision with publisher class/salience labels. Preserve parents/offsets, source credits and cross-corpus links. Short event-class clips are not certified continuous drone-absent test hours. Original description specifies CC BY-NC 3.0. |
| [TUT2017 evaluation](https://zenodo.org/records/1040168) | Verified metadata: 1,620 ten-second clips, 109 original recording IDs, **4.5 represented hours**, no overlap within parent intervals. Five outdoor classes contribute 1.5 h. | Real background scenes. Publisher scene labels support auxiliary training; recording/site independence and drone-absence evidence need assessment for stronger claims. Keep original mapping and custom noncommercial research EULA. |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | Official CSV: 2,000 five-second clips, 1,524 Freesound parent IDs, 2.7778 nominal file-hours. Original offsets are absent from this CSV. | Already consumed development/confuser material. No fresh blind-test claim. Join Al-Emadi derivatives and UrbanSound shared parents. ESC-50 is CC BY-NC 3.0; ESC-10 has separate CC BY terms and per-clip credits. |
| [DNC](https://doi.org/10.14279/depositonce-11645) | Original record/README: 4,377 mechanic/melodic/quiet recordings collected through browser audio APIs on Windows/Mac. DepositOnce API pins DNC.zip to 4,212,000,459 bytes and MD5 9b74793cd896b0e0c78854eb3a551486. Actual duration is unmeasured here. | Potential background/confuser variety; not automatically outdoor coverage. Inspect archive metadata, recorder/session grouping and native formats. The original audio record itself links MIT; this was checked independently of the code repository. |

Quality is assessed against a use: an onboard dataset can be valuable for rotor
acoustics while providing little evidence about distant ground detection.
Source labels are useful training evidence with stated limitations. They do not
all require personal relabelling by the user, nor do they all support continuous
false-alarm or audible-event evaluation.

## Additional already identified material

- **[TUT2017 development](https://zenodo.org/records/400515)** adds 4,680 clips,
  248 original recording IDs and **13 represented hours**. Its parent IDs are
  disjoint from evaluation in the inspected metadata. Both releases together
  provide 17.5 represented hours across 357 recording IDs; the five outdoor
  scenes total 5.8333 h. Recording IDs are not automatically distinct sites or
  capture days. This expands beyond the evaluation subset listed by the user.
- **[RealMAN raw background](https://github.com/Audio-WestlakeU/RealMAN)** has
  144.4167 h in rounded publisher scene metadata, including 28.45 outdoor and
  16.65 semi-outdoor hours. This is already a substantial background candidate,
  separate from its loudspeaker speech data. The earlier selective native-FLAC
  transfer proves one member's access, not every archive's extraction layout.
  Audit the remaining archives and use one linked microphone view per timeline.
- **Sydney Air 3, IDMT Berne, Polish UAV/speech and NEAPTIDE** remain in the
  [expanded registry](../../data/corpus-candidates-v2.json) and
  [earlier audit](../corpus-acquisition-v2/README.md). Their known dates,
  hardware, flight conditions, label limitations and open extent checks remain
  relevant. None is discarded because it individually falls below an hour quota.
- **DADS** reports about **60.91 file-hours** (27.10 drone, 33.81 background) on
  its [dataset card](https://huggingface.co/datasets/matei219/drone-audio-detection-samples). This is not
  a measured total of independent, suitable real field recordings. Reconstruct
  original paths/lineage before using any aggregate subset; mirrors and chopped
  mixtures do not create new capture conditions. No DADS Parquet was downloaded
  or scored in this audit.

Do not sum these figures into an accepted corpus total: definitions differ,
sources overlap, and labels/extent checks remain incomplete.

## Concrete overlap findings

The measured [metadata summary](metadata-summary.json) records:

- **94 shared Freesound parent IDs between ESC-50 and UrbanSound8K.** This proves
  shared original identities, not that each pair of selected excerpts overlaps
  in time. Conservatively link the parents across sources and keep their
  consumed-development status visible.
- **UrbanSound excerpt overlap:** 31,504.215497 summed seconds versus
  14,264.203667 represented seconds after interval union within each parent.
- **Al-Emadi duplicate layouts:** all 10,625 distinct WAV blob IDs appear in both
  binary and multiclass layouts. Git blob identity is exact repository content
  identity, not a measure of independent recording time. Its 20,000 matching
  unknown paths represent two layouts of five named chunks from each ESC-50 file.
- **Parent IDs crossing publisher folds:** four ESC-50 IDs and five UrbanSound
  IDs occur in more than one supplied fold in the inspected CSVs. This is a
  metadata grouping finding, not a claim to reproduce or invalidate published
  classifier results. Our split must keep each parent together irrespective of
  those fold labels. TUT development/evaluation have no shared parent IDs in
  their inspected metadata; broader location/date separation is not established.

## Data quantity and next execution

There is no measured answer yet to how much training data this detector needs.
Hours measure exposure; independent capture groups and covered conditions measure
different aspects of variety. Model initialization/capacity, label reliability,
acoustic observability, target error and condition shift all affect the amount
needed. A corpus can meet an hour quota and still confound class with recorder
or source. Conversely, useful transfer learning need not wait for 100 hours.

The [revised plan](../../docs/research-plan-v2.md) specifies nested training sets
that add whole capture groups, several group draws/seeds, common development
conditions and the same subsets for neural and non-neural baselines. Measure
recall, false episodes, occupancy and delay against both groups and time.
Continue scaling or acquire specific missing conditions according to these
curves and errors. A poor plateau requires investigation, not automatic volume
or architecture expansion. Reserve blind evaluation before constructing curves.

The next acquisition work is broad intake, not another small AuDroK-only study:

1. Use the original metadata above to prepare a resumable multi-source Drive
   intake, with declared storage sizes/checksums and one native microphone view
   where sources permit selective acquisition. Keep the completed AuDroK intake.
2. Prioritize full TUT background releases and suitable UrbanSound/DNC confusers;
   expand fixed-channel RealMAN beyond the one-minute access proof. Inspect
   source annotations and representative audio without assigning false strong
   labels or classifying every general-sound clip as independently reviewed absence.
3. In the same acquisition phase, inventory the remaining AuDroK packages and
   complete the already identified field sources' archive/label mappings. Inspect
   SPCup/DREGON/fault parentage for auxiliary use. Track matched positive/negative
   conditions and dynamic flight explicitly; background-only breadth does not
   establish positive-field coverage.
4. Freeze a grouped manifest and feasible evaluation design, then run the bounded
   baseline comparison and grouped learning curves. Do not wait for an arbitrary
   hour total; do not start another narrow model cycle instead of this intake.

**Future dataset publication is deferred**, as requested. Keep source versions,
original identities, derivative links, processing history, checksums, credits and
per-source terms now. A later curated release might distribute metadata or
retrieval instructions where source audio cannot be redistributed. No common
license, release permission or quality superiority over DADS is claimed here.

## Evidence and reproduction

[Source document/file identities](source-documents.json) record retrieval URLs,
sizes and SHA-256; selected publisher archive MD5s are recorded separately from
verified small-file transfers. Raw metadata remains under ignored
`runs/source-scope-audit-v3`, not as copied audio in Git. Retrieval was on
2026-11-10 UTC.

The UrbanSound CSV mirror exactly matches MD5 5015c37ce7d54c0bb8cd40a0b5ee2074
in the [soundata release index](https://zenodo.org/records/11176928). The index
itself matches its loader's pinned MD5 1c4940e08c1305c49b592f3d9c103e6f.
This verifies metadata identity; it does not verify the audio archive. TUT's four
small meta/doc ZIPs match publisher MD5s; extracted members were read through
ZIP CRC checking. Source durations above are interval metadata, not audio-header
or listening measurements. Existing DroneNoise/AuDroK measurements retain their
separate evidence and qualifications in the linked reports.

Retrieve the metadata files listed in `source-documents.json`, checking their
hashes; put TUT metadata/doc members below `tut-docs/<release-directory>/`.
Then reproduce the measurements with only Python's standard library:

```bash
python scripts/summarize_source_metadata_v3.py \
  runs/source-scope-audit-v3 /tmp/source-metadata-summary.json
```
