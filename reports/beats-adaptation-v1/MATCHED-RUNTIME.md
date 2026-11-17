# Matching the frozen control to the adapted runtime

The completed adapted arm is compared with the frozen control computed for the
same native **CPU, two-thread** study. Those frozen controls were already fitted
and saved before adaptation; this report copies their existing parameters and
receipts. It performs no new fitting or threshold selection.

Earlier published files contained the cached CPU/one-thread controls. Their
[immutable snapshot](https://github.com/StoneDerek/drone-detection/blob/c610759e3ac16ac9654d876a65698fe7be5ceed6/reports/beats-adaptation-v1/frozen-controls.json)
remains in Git history. The source folds, training IDs, weighting, C1 policy and
DEV operating rule are the same; the feature/runtime identities and fitted
numeric heads differ. The current paired model must use its matching control.

One headline count differs: the AuDroK-held/current frozen control has **97/7,931
FSD false positives** in the matched CPU2 study, versus 98 in the earlier cached
reference. All other headline frozen confusion counts are unchanged. This is
disclosed rather than silently presenting a changed reference as an improvement.
The adapted comparison uses 97 versus 141 false positives.

[matched-runtime.json](matched-runtime.json) records both sets of parameter hashes,
runtime thread identities, the unchanged plan hash and observed source-head file
timestamps. Published frozen parameters exactly match the study's existing files.
The paired analyzer verifies complete model identities, recomputes DEV cutoffs
and reproduces both validation confusion tables before reporting changes.
