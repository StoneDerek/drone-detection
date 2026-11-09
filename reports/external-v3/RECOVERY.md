# Recovered starting point and frozen CNN exports

The original `/workspace/drone-detection` working tree is preserved. A separate
`external-assessment-v3` worktree starts at committed source
`40bf036b3fe26eb210ec1fe68e74359b49b3134f`. Its locked environment passes the
80 pre-existing Python tests and Ruff. The earlier 66-test check concerned the
older restored working tree.

The two `coherent-frequency` frozen bundle directories referenced by the
committed catalog are absent from the restored commit tree, including a fresh
Git clone inspection. Their deployed numeric exports are available. Recovery
therefore reads the existing public per-model metadata and float32 array shards,
verifies each complete-array SHA-256, and checks the original model ID,
architecture and frozen decision against committed evaluation reports. It does
not retrain or substitute normalization arrays.

The rewrapped portable packages receive new metadata/package identities. Both
identities are retained explicitly in the frozen panel. They are not represented
as byte-identical copies of the absent original NPZ packages. Numeric weights,
feature configuration and decisions come from the existing deployment.

| Source CNN | Original model ID | Recovered package ID |
| --- | --- | --- |
| coherent-frequency-seed42 | `5668eaf22c46345685a7f58ae810835cd9d3905009f2f6c1095161b0ee8ac5da` | `028108d143f8b511e9814727f41cbb27021ec82f7d873e8336b0ca39639c0abd` |
| coherent-frequency-seed7 | `87c42c9ec3c41dc1e709acf821dc48adf8b27c04d9e80c821fff80cd0a4c3859` | `ee9dabc431bca8b5c684d7ff50cf324dbe2900957db18b6bdc3d87c72ad48091` |

Each recovered model is checked on the exact pre-existing fixture scope:
14 Batear replay windows and one context each from Svanstrom and ESC. All
32 model/context comparisons match the committed browser/Python records within
`1e-5`; maximum observed score difference is about `2.11e-7`. These are software
checks, not additional acoustic trials. The parity JSON retains each error.

Original deployed metadata, array shards, checksums and complete repackaged
models are retained under `artifacts/external-v3/recovered/`. The fixed panel
references its original execution paths under ignored `runs/external-v3/recovery/`;
the tracked artifacts preserve the same files for future recovery. Recreate that
directory by copying the two tracked model directories if the ignored execution
workspace is unavailable. YAMNet and SVM use the original existing packages.

The old development and v2 assessment remain consumed. No environment
publication, external deployment or product-default replacement occurs here.
