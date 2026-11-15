# Public-corpus study status

The full intake is verified: 222 distribution files, 39,010,710,391 bytes.
The accepted weak-label central-crop study has 20,573 examples: 9,959 training,
1,980 development and 8,634 final. Selected crop time is 16.6205 hours,
from 64.7792 summed native-file hours; these are not independent audible-event hours.

All four heads and global thresholds are fitted and frozen. Development positives
are one Polish capture date; the negative set is TUT2017 evaluation. Spectral
recall is 341/360; frozen CLAP and BEATs each achieve 359/360. The level/duration
control achieves 0/360. Each cutoff gives 16/1,620 negative-crop false positives
on its calibration set. These are development results, not final transfer results.

Final feature extraction is in progress. No final performance result is claimed
here. The final comparison will add per-source recall, negative-class false
positives, paired-domain AUC, grouped learning curves and an error-review export.

Before final use, 466 UrbanSound8K crops from 92 present parents shared with
historically consumed ESC-50 were quarantined. The reference intersection contains
94 parents; two have no retained crops. This review used provenance metadata,
not final detector scores. Every training/development row and both earlier fitted
control parameter sets remained identical.

The data have weak inherited labels and few independent positive campaigns.
Continuous false alarms/hour, event recall and detection delay are not measured.
See [the protocol](../../docs/corpus-study-v1.md) and the machine-readable
development results.
