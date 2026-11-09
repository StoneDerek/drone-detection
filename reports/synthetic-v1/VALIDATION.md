# Validation evidence, November 8, 2026

These are software and controlled-signal checks. No authentic audio evaluation,
microphone capture or fresh cloud-machine validation was possible.

| Check | Observed result |
| --- | --- |
| Locked setup, rerun | `bash scripts/setup.sh` completed; exact `uv.lock` dependencies |
| Software tests | **42 passed**, including CLI train/tune/development/infer, full-recording metrics, class mapping, temporal labels, silence, short/stereo/resampled/nonfinite input, duplicate/session leakage, artifact tampering/missing statistics, causal-prefix and irregular-block parity, final reuse guard, and demo selection binding |
| Static checks | `ruff check src tests` and `git diff --check` passed |
| Synthetic controlled experiment | Seven arms, 90 generated recordings / 30 generator session groups; 48 train, 12 development, 30 generator-held-out final files; one frozen final comparison batch |
| Dataset repeatability | Regenerated audio/audit/grouping/splits under a different root; full manifest SHA exactly matches frozen data identity, without rescoring final audio |
| Portable build | Python wheel built using pinned setuptools 80.9.0 |
| Clean inference install | Hash-checked runtime dependencies + wheel in `/tmp/drone-clean-v1`, not editable; Python `-I` from an unrelated working directory; copied model/WAV only |
| Clean inference/replay parity | **19/19 scores, alarm states, timestamps, windows, padding and reasons exactly matched**; also checked against the saved frozen reference trace |
| Cold invocation | File command 2.256 s, replay command 0.824 s including Python/import/load on this host; separately, 12 s audio processing took 0.026–0.027 s. These are two observations, not a startup-latency distribution |
| Wall-clock replay | 19 windows emitted over 12.003 s; score/state parity with offline, first context available after 3 s. This starts after loading/decoding and excludes unknown physical audio-device buffering |
| Browser demo | Chromium 156 / Playwright 1.62.1; all seven inputs present, audio decoding/seeking/playback advanced, positive showed ALARM, disclosed missed event showed NO ALARM, 19 timestamped rows, no page errors |
| Model/source integrity | Both model source fingerprints matched the preserved evaluated-source archive; artifact hashes verified |

`clean-environment.json`, `browser-check.json`, `paced-replay.json`, the screenshot,
all final score traces and the frozen comparison plan retain supporting evidence.
The screenshot deliberately shows the candidate's missed event at 8 s.

Direct `file://` navigation is administratively blocked by the managed cloud
Chromium configuration. Browser validation used a temporary loopback HTTP server,
not a user-facing preview. An ordinary local browser can open the self-contained
HTML directly. For the optional managed-image browser smoke check, start a local
server in this folder and run `tests/browser-smoke.cjs` with its internal URL,
screenshot path and result JSON path. Core Python checks need no browser.

## Independent evidence still missing

The supplied packet has 17 extracts and 23 figures but zero WAV files, weights,
training statistics or executable historical code. It inventories 1,461 WAV
filenames in the absent original archive; those counts do not establish independent
examples. S18 and Discord excerpts mentioned in the brief are absent. Native Git
access to the historical repository returned 403; the public page returned 404.
No acoustic capture device exists at `/dev/snd` in this instance.

Required input: `Drone Detection Project-20261108T015626Z-1-001.zip`, recording
provenance and labels/permissions, followed by the collection protocol's authentic
holdout and hard-negative exposure. Additional repository/model access may recover
historical baselines but is not required to continue with the new controlled
pipeline. No credential values are requested in chat.

There is **no supported authentic reliability claim**. On the synthetic holdout,
the candidate detected 9/10 events, missed one, and had zero false alarms over
0.0806 negative hours. Its conservative 95% FA-rate upper bound is still 45.79/h;
recall lower bound 0.555 and detected-event p95 delay 4.201 s fail contract targets.
Only replay runtime passes on the reported Xeon CPU. Gain invariance is a
representation property, and these tests do not identify the historical models'
failure cause. Augmentation and confirmation did not establish an additional
material benefit on development. No larger or pretrained model advantage is
claimed without verified audio.

After the frozen comparison, changes were limited to software validation/demo
guards and tests; the exact evaluated source is preserved. The holdout has not
been used for further model/threshold optimization. The failure appendix replays
the frozen system for inspection and is explicitly selected after evaluation.
Remote GitHub Actions execution and cloud publication were not performed.
