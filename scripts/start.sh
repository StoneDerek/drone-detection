#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
demo_args=()
if [[ -f runs/real-v2/corpus/manifest-v2.jsonl && -f reports/real-v2/demo-selection.json ]]; then
  demo_args=(--manifest runs/real-v2/corpus/manifest-v2.jsonl --selection reports/real-v2/demo-selection.json)
fi
exec .venv/bin/drone-detector serve --bundle artifacts/real-v2/candidate "${demo_args[@]}" "$@"
