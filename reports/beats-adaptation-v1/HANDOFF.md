# BEATs adaptation study

The four frozen-encoder controls and the first complete adapted evaluation are
finished. [The current results](RESULTS.md) and [paired assessment](paired-analysis/audrok-dresden/current/INTERPRETATION.md)
record all 7,989 prescribed validation clips: AuDroK recall increases from 31/58
to 34/58, while FSD false positives increase from 97/7,931 to 141/7,931. This is
not a clear reliability gain. Three adapted arms remain. The complete corpus has
30,238 crops; every neural arm uses three full epochs. Partial training does not
enter the comparison table. The earlier [CPU execution report](cpu-study/RESULTS.md)
and optimization audit remain dated progress snapshots.

The numerical trainer and training driver are byte-identical to immutable commit
`faeaf5e024ac65b8c28bc3bceeed31736e112b1b`. A continuation shortens CPU recovery
checkpoints from 1,000 steps to 100 steps. The explicit migration verifies the
entire model, optimizer, RNG and counter fingerprint before and after changing
only cadence metadata. The source folds, data, hyperparameters and accumulated
budget remain unchanged. The supervisor runs from an isolated published checkout.

The recovered local CPU supervisor uses published commit
`fcc140b6880d6af10ee81878df6efbe9af6558ad`. Its new coordinator verifies and reuses
completed arms, then delegates unfinished work to the unchanged numerical driver.
The [supervision recovery record](SUPERVISION-RECOVERY.md) describes the explicit
metadata migration and completed-model preservation. All three jobs passed in
[run 44141716431](https://github.com/StoneDerek/drone-detection/actions/runs/44141716431).
The GPU notebook separately remains pinned to implementation
`2a4de5723cf755ab3a9187c8c6027030cc616027`.

## Completed evidence

[RESULTS.md](RESULTS.md) contains all four frozen controls, one completed adapted
row and their limitations. [Runtime provenance](MATCHED-RUNTIME.md) explains the
matched CPU2 controls and the earlier published CPU1 references.
The exact fitted heads, operating thresholds and source-level counts are retained
alongside it. Broader positives did not consistently improve transfer. These
known-source development results do not identify the single cause of failure.

Current software validation: 220 standard checks passed with nine optional skips.
The native training/recovery, isolation, coordination and reporting selection
passed 24 tests with two optional native-asset skips. Earlier, eleven
native BEATs/recovery checks passed, including an actual encoder weight update,
full driver completion and exact interrupted-optimizer recovery. The synthetic
signals in software checks are not a drone performance evaluation. All three
GitHub Actions jobs passed for the implementation commit:
[run 43814505622](https://github.com/StoneDerek/drone-detection/actions/runs/43814505622).
The continuation's recovery/reporting checks passed 13 tests with two optional
native-asset skips. A cadence-change test preserves the exact uninterrupted
training trajectory, and reporting tests reject stale process state, changed
classifiers and incomplete epochs. The full real checkpoint migration also
verified the unchanged non-identity training fingerprint.

## GPU execution from the original archives

Open [the complete GPU notebook](../../notebooks/beats_adaptation.ipynb) in Colab,
select a GPU runtime, check the two Drive folder settings and run all cells.
The notebook pins the implementation commit above. The original 222-file intake
is sufficient: no old model, embedding archive or completed comparison notebook
is required. Missing inputs fail explicitly; there is no reduced-cohort fallback.

The first run checks all publisher distribution digests, prepares the complete
corpus and acquires the additional public challenge audio. Canonical audio and
checkpoints persist in a separate Drive project. Verified local waveform copies
avoid repeated training reads over Drive. Completed preparation and models are
reused after a runtime replacement. A GPU project is separate from the local CPU
study; its frozen controls are recomputed on its own runtime.

The notebook schema and every code cell have been validated. GPU execution has
not been exercised in this CPU-only environment. The pinned Colab page now opens
in the restarted shared browser, but that browser is signed out. No remote GPU
run has started; the local CPU study continues.

## Local supervision and recovery

The local study directory is `work/beats-adaptation-cpu2-v1`. Its authoritative
current state is `pipeline-state.json`; `supervisor.log` identifies the active
stage. Stage logs are `native-base-embeddings.log`, `paired-frozen-controls.log`
and `four-full-fine-tuning-arms.log`. Training checkpoints are under each
`<held-source>/<positive-mix>/fine-tuned` directory. Full results will be written
to `fine-tuned-controls.json` only after all four arms complete.

The initial 30-hour estimate from repeated synthetic batches was too optimistic.
The actual first training segment averaged 8.22 seconds/batch: extrapolation to
30,864 training batches is about 70 hours of training alone, before selection,
feature extraction or interruption replay. This is an estimate, not a measured
completion time. Completing all four CPU arms is not expected inside the cap at
that rate. The GPU notebook is the practical route to the full comparison.
The supervisor retains the cumulative 44-hour budget and requests a cooperative
checkpoint pause rather than reducing data or epochs. Saved elapsed time carries
across restarts. No automatic follow-up notification or scheduled run is configured.

The environment replacement interrupted the first epoch after 1,675 logged
steps. Recovery verified the 1,000-step checkpoint and replayed unsaved batches.
The replayed logged losses and gradient norms matched the original trace. The
shorter checkpoint interval limits future replay; it does not extend the budget.

On November 17 the second arm was also found interrupted, with step 3,400 safely
checkpointed and step 3,425 last logged. The cause of this interruption is
unconfirmed. Supervision recovery retained exactly 98,534.66488816802 cumulative
seconds, reused the first complete result and resumed the second arm's saved
optimizer, RNG and next batch. No neural checkpoint migration was necessary.

To request a cooperative pause, create `pause-requested.json` in the study
directory. Wait for the supervisor to report `paused` before replacing a runtime.
Only one writer may use a study directory. An interrupted training arm resumes
its saved optimizer, RNG and exact next batch; feature extraction restarts if it
was interrupted before the completed matrix was saved. Preserve the pause marker
and inspect its reason before deliberately resuming or extending a budget.

## Interpretation after completion

Compare each adapted arm with its matching frozen control at the same declared
selection rule. Report WSU drone recall and ambient-tail errors, AuDroK recall,
and FSD false positives separately. Improvements confined to one collection do
not establish generalization. WSU/AuDroK presence labels have not received blind
human audibility review, and many windows share few recording campaigns.

ITU ARIS is excluded from this BEATs study, but was already evaluated and used
for within-session learning earlier in this project. It is not an untouched
project-wide holdout. Its v1.1 documentation update preserves the v1.0 audio.
The corrected [source reservation](future-source-reservation.json) distinguishes
this study's actions from prior project exposure. The
[new source audit](../positive-source-audit-v1/README.md) records prospective
field/indoor candidates and their outstanding checks. DADS also contains already
used sources and is not a fresh test collection as a whole.

The [complete FPV lineage audit](../fpv-lineage-v1/README.md) now verifies the
candidate FPV ZIP and accounts for all 19,286 outer audio members. Its publisher
splits share capture groups and many exports lack an included declared parent.
The candidate remains unscored and outside this experiment; it is not yet a
qualified independent confirmation set.

The [optimization/coverage audit](optimization-audit/README.md) records the first
arm through two complete epochs and a step-5,600 recovery snapshot. Frozen TRAIN
loss was already near zero while selection loss was substantially higher. Full
schedule coverage, protocol weight allocation, checkpoint finiteness and starting
validation arithmetic passed the audited checks. The final C1 refit and completed
adapted source-level evaluation were still pending at that snapshot.

The [complete DADS metadata census](../dads-source-census-v1/README.md) covers all
180,320 path/label rows. Every stored filename is a class-and-sequence identifier;
the shipped metadata does not preserve original source/capture fields. This audit
does not add DADS audio to training or resolve its recording independence.
