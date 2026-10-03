#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
mkdir -p work/tmp work/cache work/config
export TMPDIR="$ROOT_DIR/work/tmp" TEMP="$ROOT_DIR/work/tmp" TMP="$ROOT_DIR/work/tmp"
export CUDA_CACHE_PATH="$ROOT_DIR/work/cache" XDG_CACHE_HOME="$ROOT_DIR/work/cache"
export PIP_CACHE_DIR="$ROOT_DIR/work/cache" XDG_CONFIG_HOME="$ROOT_DIR/work/config"
export PYTHONDONTWRITEBYTECODE=1
if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
fi
if [[ ! -f .venv/unleashed-dependencies-v1 ]]; then
    .venv/bin/python -B -m pip install --disable-pip-version-check -r requirements-unleashed.txt
    touch .venv/unleashed-dependencies-v1
fi
exec .venv/bin/python -B unleashed.py "$@"
