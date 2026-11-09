#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/drone-uv-cache}"
uv sync --frozen --extra dev --extra pretrained
