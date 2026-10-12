#!/usr/bin/env bash
# Build tensorfold-native (with the CUDA fatbins embedded) from kinetick-labs/TensorFold on a DGX Spark.
set -euo pipefail
TF_DIR="${TF_DIR:-$HOME/TensorFold}"
TF_REF="${TF_REF:-perf/zig-exl3-decode}"
ZIG="${ZIG:-$(command -v zig || echo "$HOME/zig-0.17.0/zig")}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
if [[ ! -d "$TF_DIR/.git" ]]; then
  git clone https://github.com/kinetick-labs/TensorFold "$TF_DIR"
fi
cd "$TF_DIR"
git fetch origin "$TF_REF"
git checkout --detach FETCH_HEAD
echo "[build] $(git log -1 --format='%h %s')"
"$ZIG" version | grep -q '^0\.17\.' || { echo "[build] Zig 0.17.x required (.zig-version)" >&2; exit 2; }
# -Dnvcc compiles our .cu kernels (fn_exl3 & co.) for sm_121 and embeds them in the binary.
"$ZIG" build native -Dnvcc="$NVCC" -Dsm=121
echo "[build] -> $TF_DIR/zig-out/native/bin/tensorfold-native"
