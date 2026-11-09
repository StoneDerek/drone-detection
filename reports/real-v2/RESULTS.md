# Real-audio product assessment v2

Recording-level results against publisher labels. No independently verified event timing or field-reliability claim.

The first real assessment failed and was explicitly consumed for development. It was not reused as a fresh test. See docs/real-data-investigation.md.

## Data

| Split | Files | Positive files | Negative files | Provenance/day groups | Audio hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 405 | 55 | 350 | 237 | 1.912 |
| development | 210 | 22 | 188 | 145 | 1.908 |
| final | 1741 | 133 | 1608 | 1222 | 3.380 |

File counts are not independent trials. DDL excerpts share their original flights and capture day; the Batear day is one group. The small number of positive capture days limits uncertainty and operating-range claims.

## Frozen comparison

Selected on development: **yamnet**, model `b9c7df3a56936a161dbf7e70fb02b6979666bfe8043d131a4e38496da13c261f`.

| Arm | Positive files detected | False-positive negative files | Recall | Negative-file FPR | Precision | Real-time factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| relative | 67/133 | 109/1608 | 0.504 | 0.068 | 0.3806818181818182 | 0.0041 |
| yamnet | 100/133 | 126/1608 | 0.752 | 0.078 | 0.4424778761061947 | 0.0317 |

## Selected model by source

| Source | Files | Positive files detected | False-positive negative files |
| --- | ---: | ---: | ---: |
| batear-new-day | 55 | 19/43 | 2/12 |
| ddl-real | 90 | 81/90 | 0/0 |
| esc50-fresh-negatives | 1596 | 0/0 | 124/1596 |

## Limits and failures

Frozen decision: `{'off_hits': 2, 'off_threshold': 0.375, 'on_hits': 2, 'on_threshold': 0.5}`. Group-bootstrap intervals: `{'false_positive_fraction': [0.06308514538934694, 0.09348137273942399], 'recall': [0.4418604651162791, 0.9], 'warning': 'Conservative source/day groups, not independent windows. Few positive-source groups make intervals inadequate for field claims.'}`.

The provisional recording targets are recall at least 0.9 and negative-file FPR at most 0.05. Observed checks: recall **FAIL**, negative-file FPR **FAIL**. These point checks do not fulfill the original event recall/FA-per-hour/delay contract, which remains unassessed.

Observed errors (all are retained in JSON; first examples below):

- `batear-new-2474750763FAA288_20260428_072320` / batear-new-day: publisher label drone; no alarm; maximum score 0.3485.
- `batear-new-2474750763FAA288_20260428_072430` / batear-new-day: publisher label drone; no alarm; maximum score 0.6237.
- `batear-new-2474750763FAA288_20260428_072800` / batear-new-day: publisher label non_drone; alarm; maximum score 0.8942.
- `batear-new-2474750763FAA288_20260428_073020` / batear-new-day: publisher label drone; no alarm; maximum score 0.7812.
- `batear-new-2474750763FAA288_20260428_073350` / batear-new-day: publisher label drone; no alarm; maximum score 0.5455.
- `batear-new-2474750763FAA288_20260428_073720` / batear-new-day: publisher label drone; no alarm; maximum score 0.5116.
- `batear-new-2474750763FAA288_20260428_074310` / batear-new-day: publisher label drone; no alarm; maximum score 0.1723.
- `batear-new-2474750763FAA288_20260428_074420` / batear-new-day: publisher label drone; no alarm; maximum score 0.0961.
- `batear-new-2474750763FAA288_20260428_074530` / batear-new-day: publisher label drone; no alarm; maximum score 0.4961.
- `batear-new-2474750763FAA288_20260428_074640` / batear-new-day: publisher label drone; no alarm; maximum score 0.5519.
- `batear-new-2474750763FAA288_20260428_074900` / batear-new-day: publisher label drone; no alarm; maximum score 0.0705.
- `batear-new-2474750763FAA288_20260428_075010` / batear-new-day: publisher label drone; no alarm; maximum score 0.1731.
- `batear-new-2474750763FAA288_20260428_075120` / batear-new-day: publisher label drone; no alarm; maximum score 0.1178.
- `batear-new-2474750763FAA288_20260428_075230` / batear-new-day: publisher label drone; no alarm; maximum score 0.0180.
- `batear-new-2474750763FAA288_20260428_075340` / batear-new-day: publisher label drone; no alarm; maximum score 0.0553.

Class labels describe recordings, not independently listened-to drone events. Flight metadata and GPS distance do not establish acoustic audibility. No supported microphone, distance or SNR range is claimed. Event recall, false alarm events per true negative hour, physical capture delay and live-microphone operation require separately annotated capture.

Next useful work: independent full recordings from additional days/devices, with audible onsets/offsets and long hard-negative sessions. Keep this assessment consumed if its errors guide further model optimization.

The app supports real file upload, full-recording scores, playback/seeking, alarm intervals, input-quality flags and JSON export. It is recorded replay. The demo includes metadata-selected positives/ordinary/hard negatives and a labeled observed-error appendix.
