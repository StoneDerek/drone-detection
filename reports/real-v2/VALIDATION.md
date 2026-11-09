# Validation of the real-audio product

This iteration ships a runnable prototype. It **does not** meet its reliability targets.
The frozen fresh assessment detected 100/133 publisher-positive files and alarmed
on 126/1608 publisher-negative files. These are files, not independent trials.

- **49 software tests passed**: decoding, labels, grouped splits, missing/tampered
  artifacts, train-only statistics, weak supervision, final-use guard, causal
  buffering, short/silent inputs, file/replay parity, API uploads/errors and examples.
- Locked setup, static checks, wheel build and repository whitespace checks passed.
- The wheel was installed with hash-checked dependencies into a second virtual
  environment. Python `-I` ran from an unrelated directory with a copied model
  and audio only; no editable project or notebook state was used.
- All 14 inference/replay windows agreed in scores (observed maximum
  error 0, tolerance 1e-5), alarm states, context
  timestamps, padding and input reasons. Model identity matched the frozen package.
- Paced replay took 9.106 s for 9.078 s audio **after** 5.903 s encoder
  initialization; p95 delivery lag after initialization was 58.5 ms. Hardware/OS
  capture buffering and true drone-event detection delay remain unmeasured.
- Cold file invocation took 18.065 s including imports/loading; its processing
  timer was 14.118 s. A first cold pretrained invocation is not a real-time claim.
  The web service warms its model before accepting requests; CLI replay warms
  before starting the playback clock. This startup fix does not change scores.
- Frozen-assessment steady processing real-time factor was
  0.0317 on INTEL(R) XEON(R) PLATINUM 8573C.
  See JSON for inclusions and host details; this is same-host measurement.
- Chromium browser checks passed real upload, playback/seek, timestamped rows,
  alarm display, JSON export and actionable invalid-audio handling with no page
  errors. An observed missed case was then replayed from the frozen model with a
  visible failure-appendix label and **No alarm**. Screenshots retain both cases.
- A standalone server from the isolated wheel and packaged model passed a real
  upload/inference request **without the evaluation corpus** or any examples.
- Both model packages' source fingerprints match the preserved evaluated source
  archive; SHA-256 checks cover all delivered model/dependency/encoder files.

[Product screenshot](product-screenshot.png) · [Observed missed case](failure-screenshot.png)

The app is local recorded replay. No physical microphone or new cloud-machine
restoration test is claimed. No external deployment or environment publication
was performed. Live processes are not a reusable environment snapshot.

## Why this is not a reliable field detector yet

DDL contributes 90 correlated excerpts from 30 flights on one new capture day;
Batear contributes 43 publisher-positive clips from another day. Grouping treats
these days conservatively. Published physical presence and distance do not prove
acoustic audibility throughout. Group confidence intervals are wide, and both
point targets (90% recall, at most 5% negative-file FPR) fail.

The original event-recall, true FA/hour and delay contract remains unassessed
without independent temporal/audibility annotation and long reviewed negative
sessions. The DDL source had 81/90 detected; Batear had 19/43. Generalization
across microphones/conditions is still a material limitation.

False alarms include vacuum cleaners (25), sheep (17), trains (10) and clock
alarms (9). Those are useful hard-negative review candidates, not a reason to
retune on this assessment and call it independent again. The next experiment
needs independently reviewed audible events, matched negative recordings on the
same microphones, and additional capture days. Larger training runs alone do
not supply that evidence.

Evidence is in the adjacent JSON files, manifest snapshot, data/source plan,
quality exclusions, and compressed archives containing every frozen score trace.
The failed first real assessment is retained. Earlier synthetic checks are not
counted as acoustic reliability evidence.
