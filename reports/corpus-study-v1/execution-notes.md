The initial CPU crop preparation and training/development feature extraction
used the exact source snapshots in `executed-source`. The released runner adds
explicit CPU/CUDA selection, immutable source snapshots, and a completion rescan
for incremental input parts. CPU waveform processing and the frozen encoder
computation are unchanged. Per-vector receipts retain their actual source hashes.
The final CPU extraction uses the released runner. Source/capture groups and
previous exposure are decided without final scores.
