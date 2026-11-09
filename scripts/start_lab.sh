#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
lab_catalog=reports/spectral-lab-v1/catalog.json
if [[ -f reports/cnn-foundations-v1/catalog.json ]]; then
  lab_catalog=reports/cnn-foundations-v1/catalog.json
fi
exec .venv/bin/drone-detector lab --catalog "$lab_catalog" "$@"
