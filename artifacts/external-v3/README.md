# Frozen model recovery and external execution

`recovered/` contains the two existing frequency-preserving CNNs recovered from
their deployed numeric exports, including the original exported metadata/shards
and portable repackaged NPZ/JSON models. No numerical weights, preprocessing or
decisions were trained or tuned during recovery. Original and recovered package
identities are mapped in `recovery.json` and the external assessment report.

`execution-source.tar.gz` preserves the exact source snapshot made before the
external assessment. `recovery-execution.py` preserves the recovery procedure.
SHA256SUMS covers every retained file. The model/source license and authorship
remain those of the project; third-party encoders stay separately attributed in
their original packages. No publisher audio is included here.

Results: `reports/external-v3/RESULTS.md`.
