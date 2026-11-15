# Full-corpus Colab handoff

[Open the full-corpus notebook](https://colab.research.google.com/github/StoneDerek/drone-detection/blob/1066ef6e9dde839d456d5f18fb097e77d6e99603/notebooks/public_corpus_comparison.ipynb).

The notebook fetches implementation commit
`38313a91d033919a0ee76e2fb1d8a4dc269cfb4c`, never a moving branch head.
It requires the complete pinned source intake and processes every accepted
fixed crop under the recording/source holdout protocol. There is no small-example
fallback. Input archives remain read-only; results and numerical feature
checkpoints use a separate Drive project. The default is explicit CUDA inference,
with a separately selected CPU option. GPU execution has not been verified on
this local CPU-only machine.

The local CPU study verified all 222 distributions, froze the accepted cohort
and all four models, and completed the full final batch. [Results](RESULTS.md) and
[the protocol](../../docs/corpus-study-v1.md) state its scope. The handoff notebook
passes nbformat validation and compilation of all code cells. Local lint and
181 software tests pass; eight optional checks are skipped.

Results describe weak-label central-crop classification, not continuous-stream
event recall or false alarms per hour. Dataset terms and pretraining provenance
limitations remain as documented.
