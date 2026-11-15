# Broader original-source intake, 2026-11-14

**A resumable 39.011 GB original-source intake is ready for the Drive-connected
CPU notebook.** It expands beyond the completed AuDroK subset. Full transfer has
not run in this Codex workspace, which lacks bulk space and authenticated Drive
write access. Distribution bytes are not independent hours or accepted training
data. No model was trained or scored.

[Open the broader intake notebook](https://colab.research.google.com/github/StoneDerek/drone-detection/blob/main/notebooks/corpus_intake.ipynb).
Keep the completed AuDroK source folder and receipts. This notebook uses a new
project folder and code pin, so earlier completed intake/recovery state remains
usable. Select CPU; no GPU or additional packages are needed.

## Selection and storage

The [frozen intake manifest](../../data/corpus-intake-v3.json) contains 222 unique
distribution paths across eight release entries. Every file has a pinned
publisher MD5 and expected length. Source-specific terms, metadata response
hashes and label limitations are retained. Releases include multiple views,
calibration, documents and processed measurements; these are preserved for
inspection rather than assigned invented labels.

| Release | Files | Distribution bytes | Relevant qualification |
| --- | ---: | ---: | --- |
| TUT2017 development | 13 | 10,700,420,548 | Metadata supports 13 represented hours; scene labels, original recording IDs. |
| TUT2017 evaluation | 6 | 3,599,577,336 | Metadata supports 4.5 represented hours; preserve parent mapping across excerpts. |
| UrbanSound8K | 1 | 6,023,741,708 | 8.75 excerpt-hours cover 3.96 represented source-hours; cross-links to ESC-50. |
| DNC | 1 | 4,212,000,459 | Browser/computer-recorded background classes; unique time/groups unresolved. |
| IDMT Berne | 1 | 1,515,112,211 | Array recordings and flight logs; reconstruct excerpt parentage. |
| Polish UAV/speech | 19 | 12,045,473,772 | Hovering measurements, repeated models, two recorder types; not all moving-flight coverage. |
| NEAPTIDE | 7 | 176,260,894 | Segmented/propagation-corrected measurements; retain documented calibration. |
| DroneNoise v3 | 174 | 738,123,463 | One campaign, repeated microphones; duplicate publisher entry retained as an alias. |

| Profile | Files | Bytes | Purpose |
| --- | ---: | ---: | --- |
| `core` | 222 | 39,010,710,391 | All selected field/background distributions and supporting files |
| `field` | 201 | 14,474,970,340 | The four field-measurement candidates, including processed measurements |
| `background` | 21 | 24,535,740,051 | Both TUT releases, UrbanSound8K and DNC |
| `metadata` | 9 | 6,216,596 | Small annotation/document packages only |
| `verification` | 3 | 37,547,419 | Technical transfer check, not a training corpus |

The notebook defaults to `core` and declares a 60 GiB allocation. Set that value
from available Drive space before running; it is not a quota measurement. Core
archives require 36.332 GiB in addition to existing files. Do not extract all
archives onto notebook SSD or assume that Drive capacity enlarges runtime disk.
Native caches will be prepared source by source after inventory/label checks.

RealMAN bulk single-channel extraction and remaining AuDroK packages are still
separate acquisition work. The previously demonstrated one-minute RealMAN
member does not establish every archive layout. This selection does not claim
to exhaust suitable public data, satisfy independent positive-field coverage
or outperform DADS.

## Recovery and integrity

[`corpus_intake.py`](../../src/drone_detector/corpus_intake.py) and its
[CLI](../../scripts/fetch_corpus_intake.py) retain a stable `.part` file and
identity journal. A replacement process resumes from the stored byte count with
an exact HTTP range. Wrong range/length responses are rejected before writing.
If a server ignores Range and returns a full response, that file restarts.

Completion requires the stored bytes to match pinned size/checksums. A full
SHA-256 is also recorded. Modified partials or changed server objects cannot
silently pass this check. Pending/completed receipts allow recovery around the
final rename. Completed files are rehashed before reuse. An unrecognized or
failed file is preserved for review; it is never silently accepted.

A local process lock prevents simultaneous writers to the same destination on
one runtime. It is stored on local disk, without assuming that Drive implements
POSIX locks. **One active runtime per intake folder remains required across
machines.** SIGKILL releases the local lock, so a replacement process can resume.
Flush Drive before planned shutdown and verify on an actual replacement runtime;
local read-back does not independently prove cloud synchronization.

## Verification and source findings

- Live bounded probes returned exact HTTP206 resume ranges for TUT, DNC,
  UrbanSound8K, Poland and NEAPTIDE. This verifies those endpoint probes, not
  every future request or every file. [Range evidence](range-probes.json).
- A real HTTP test killed the downloader mid-stream, started a fresh process,
  verified the exact resumed offset and final bytes/checksums, then checked
  completed-file reuse without another request. It also rejected an overlapping
  local writer before it made a request.
- Other tests cover changed identities, corrupt prefixes, unrecognized files,
  wrong resume ranges, truncated bodies, range-ignoring servers and completion
  recovery. The real manifest's unique paths and profile totals are checked.
- Live complete-file validation uses a TUT metadata ZIP, one DroneNoise WAV
  and the complete NEAPTIDE Mavic ZIP. [Receipt/native-format evidence](live-verification.json)
  distinguishes whole-file publisher checks from header/member inspection.
  All 49 WAV members in that ZIP passed CRC and native-format inspection; they
  are mono 44.1 kHz PCM32, while the DroneNoise sample is mono 50 kHz float32.
  No normalization, listening judgment or detector scoring was applied. A
  fresh CLI invocation rehashed and reused all three completed files.

An initial local reuse check was accidentally started before its first transfer
finished. Both size checks rejected the overlapping NEAPTIDE partial; no
completion was published. Its file/journal were preserved under ignored
`runs/corpus-intake-v3/rejected-overlap`. The local lock and overlapping-process
test were added before retrying. This is an acquisition failure and repair,
not acoustic evidence.

The DroneNoise release contains two entries named
`Ed_3p_10_H00_N_C_nw_ev1_M5.wav`, with identical declared length and publisher MD5,
under file IDs 46212081 and 46212195. The manifest downloads one and preserves
the second URL in `publisher_alias_urls`. The 175-entry publisher release is
therefore 174 unique selected paths here; repeated microphones remain linked
separately. The earlier file-level inventory is preserved as such.

## Execution and subsequent work

The notebook first prints the selection, then downloads to its separate Drive
folder. Rerun the same notebook/folder after interruption. It pins an exact code
commit on first use and reuses that commit on replacement runtimes. Do not change
the manifest in place for an existing intake identity.

Equivalent Linux CLI usage:

```bash
python scripts/fetch_corpus_intake.py \
  --profile core --output /path/to/durable/corpus-intake-v3 --plan-only
python scripts/fetch_corpus_intake.py \
  --profile core --output /path/to/durable/corpus-intake-v3 --budget-gib 60
```

After full acquisition, inventory archive members and native formats, recover
recording/channel/derivative parentage, check label support and inspect
representative audio. Reserve grouped evaluation before model fitting. Then
prepare bounded native caches and run the planned neural/non-neural comparison
and grouped learning curves. There is no 100-hour readiness gate and no new
detector-quality result from acquisition alone. Future dataset publication
remains deferred; original source terms and credits remain attached.
