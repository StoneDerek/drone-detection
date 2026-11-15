# Completed public-corpus study

The full CPU comparison is complete. Read [the findings and next milestone](FINDINGS.md)
and [the complete results](RESULTS.md).

- Verified acquisition: 222 distribution files, 39,010,710,391 bytes.
- Accepted cohort: 20,573 fixed crops; 9,959 training, 1,980 development, 8,634 final.
- Selected crop time: 16.6205 hours from 64.7792 summed native-file hours. These are
  not independent audible-event hours.
- Four frozen classifiers evaluated together at their fixed global cutoffs.
- BEATs: 360/360 Polish, 15/18 Edzell and 35/35 processed NEAPTIDE positive crops;
  1,329/8,221 false-positive UrbanSound8K crops (16.17%).
- Source/campaign correlation, weak labels, source/class confounding and unknown
  pretraining overlap remain material limits. Continuous false alarms/hour are unmeasured.

Before final use, 466 UrbanSound8K crops from 92 present parents shared with
historically consumed ESC-50 material were quarantined. The reference intersection
contains 94 parents; two have no retained crops. This review used provenance
metadata before any final scoring. All 11,939 training/development rows and the
two earlier control parameter sets remained identical.

[The full-corpus Colab handoff](HANDOFF.md) pins implementation commit
`38313a91d033919a0ee76e2fb1d8a4dc269cfb4c`. Local lint and 181 software tests passed
with eight optional skips. Both published GitHub CI jobs passed. The final
cohort is now consumed; subsequent tuning requires a different independent
evaluation for a new generalization claim.
