# CNN diagnosis evidence

Three diagnostic normalization models trained from the same real train/development manifest. Status is deliberately **trained**, not frozen; no alarm thresholds were selected. None is deployed or replaces a default.

See ../../reports/cnn-diagnosis-v1/RESULTS.md and normalization-comparison.json for the protocol, per-record scores, losses and limitations. The three bundles are byte-identical copies from the completed runs. Load with the included current source, not the earlier frozen0.3 wheel (which predates the explicit reference-amplitude fields).

source.tar.gz contains the Python project/dependency lock, exact source and diagnostic/observation scripts. source.json records file hashes, hardware and numeric library versions. The corpus manifest/splits are preserved in ../../reports/real-v2/manifest-snapshot.jsonl; raw audio and caches are not redistributed here. SHA256SUMS checks this packet.

The artificial architecture-control image pair is a software diagnostic only. The sixteen-real-context memorization check is a training/gradient check only. Neither is acoustic field reliability evidence.
