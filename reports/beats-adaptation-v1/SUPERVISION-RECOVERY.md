# Interrupted CPU study: supervision recovery

The second arm (AuDroK-held / broader positives) was interrupted after logging
step 3,425. Its last verified recovery checkpoint contains step **3,400**,
one completed epoch, and next batch position 679 in epoch two. The old trainer
and supervisor processes are absent. The cause of interruption is unconfirmed.
The first arm is complete; neither WSU-held arm has started.

## Recovery contract

The recovery changes coordination and supervision only. The numerical driver
and trainer remain byte-identical to implementation commit
`faeaf5e024ac65b8c28bc3bceeed31736e112b1b`:

| Component | SHA-256 |
|---|---|
| Numerical driver | `201d6f2e6be1494c203c5586cd2dec64fa19cfc350fbec72f774e1e5aa835f92` |
| Numerical trainer | `c8fc62ae8085b52073a67cb662f22a908042f020cfac038a9d5059b50750c419` |
| Interrupted step-3,400 checkpoint | `7c390aea0a660000e57c396c84d27968943bb88062330f50922b782c0ab05451` |

The arm coordinator validates a completed arm's head, selected encoder, model,
training history, plan and validation receipt before reusing it. It delegates
unfinished arms to the original numerical driver. This avoids refitting or
rewriting a completed classifier during recovery. A malformed completed result
fails verification rather than being silently retrained.

The explicit `--migrate-legacy-supervision` option accepts only the known previous
launcher, an inactive study without a pause request, and an identical plan,
device, thread count, checkpoint cadence and budget. It records the before/after
supervision metadata. It does not open or rewrite neural checkpoint files.
Exact process-argument matching and the existing exclusive writer lock guard
against a second study writer.

Saved cumulative runtime is **98,534.66488816802 seconds (27.37074025 hours)**.
Recovery retains the original **44-hour** cap, leaving approximately **16.63
hours** before additional runtime. It does not promise completion of all four
arms inside that cap. The native frontend, three epochs, sample cohort, folds,
optimizer, selection rule and cutoffs are unchanged. Unsaved batches are replayed
from the checkpoint; they do not add epochs or evaluation evidence.

## Completed model preservation

Before recovery, the selected first-arm encoder, exact final head/scaler,
threshold, training and validation receipts, official BEATs source/license,
encoder configuration and project restoration sources were archived locally.
Every archive entry was checked by SHA-256, and all **90,311,792** encoder
parameters were strictly restored from the archived weights. No audio was scored
in this restoration check. Source artifact hashes and modification times were
unchanged.

[The preservation receipt](model-preservation.json) records the archive digest.
The 336,584,678-byte ZIP remains in the local workspace; it has not been uploaded
to durable off-machine storage. The public GitHub receipt is not a weights backup.

## Software verification before restart

The standard suite passed **220 tests**, with nine optional skips. The native
training/recovery, source-isolation, coordination and reporting selection passed
**24 tests**, with two optional native-asset skips. The project-scoped Ruff check
and whitespace check passed. Coordination tests cover reuse of completed arms,
failure on invalid completed artifacts, pause handling, and migration rejection
when the runtime, protocol or cumulative budget changes. These are software
checks, not acoustic classification results.

The actual restart and post-recovery preservation checks will be recorded
separately after the published recovery implementation is running.
