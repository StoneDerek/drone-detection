# BEATs adaptation study

The four frozen-encoder controls and the complete native CPU feature matrix are
finished. Full-backbone training has started, but no adapted evaluation is complete.
[The CPU execution report](cpu-study/RESULTS.md) separates live, interrupted and
completed work. The complete cohort is 30,238 crops. Every neural arm uses three
full epochs; partial training does not enter the comparison table.

The numerical trainer and training driver are byte-identical to immutable commit
`faeaf5e024ac65b8c28bc3bceeed31736e112b1b`. A continuation shortens CPU recovery
checkpoints from 1,000 steps to 100 steps. The explicit migration verifies the
entire model, optimizer, RNG and counter fingerprint before and after changing
only cadence metadata. The source folds, data, hyperparameters and accumulated
budget remain unchanged. The supervisor runs from an isolated published checkout.

The current supervision/reporting implementation is
`2a4de5723cf755ab3a9187c8c6027030cc616027`; the current GPU notebook pins that exact
commit. All three jobs passed in
[run 43909773186](https://github.com/StoneDerek/drone-detection/actions/runs/43909773186).

## Completed evidence

[RESULTS.md](RESULTS.md) contains all four frozen controls and their limitations.
The exact fitted heads, operating thresholds and source-level counts are retained
alongside it. Broader positives did not consistently improve transfer. These
known-source development results do not identify the single cause of failure.

Software validation: 191 standard checks passed with nine optional skips. Eleven
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
not been exercised in this CPU-only environment. A second Colab navigation attempt
failed with `Browser Use could not read requirements`; no remote GPU run is claimed.

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

ITU ARIS is reserved as a possible future whole-source holdout; only metadata
has been inspected. Its single continuous session limits the independence of
its many overlapping clips. The DADS aggregate contains already used sources
and is not a fresh test collection as a whole.
