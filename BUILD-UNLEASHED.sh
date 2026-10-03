#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
mkdir -p work/tmp work/cache
export TMPDIR="$ROOT_DIR/work/tmp" CUDA_CACHE_PATH="$ROOT_DIR/work/cache"
# CUDA_ARCH may be native for this PC, or a CMake list such as '89;90' for distribution.
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
    -DSTRATA_PORTABLE=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_BUILD_TESTS=OFF \
    "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-native}" \
    "-DSTRATA_GGML_DIR=$ROOT_DIR/third_party/llama.cpp" "$@"
cmake --build build --target strata native_expert_parity kv_mixed_parity ple_q8_parity --parallel "${BUILD_JOBS:-8}"
