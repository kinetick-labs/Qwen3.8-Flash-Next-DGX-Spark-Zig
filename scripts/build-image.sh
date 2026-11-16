#!/usr/bin/env bash
# Build the runtime image serve.sh runs (image/Dockerfile): NVIDIA PyTorch 26.07 (CUDA 13) + the vision helper's
# Python package + the sm_121 Triton kernel set, compiled here and checked cubin by cubin. No GPU needed; under a
# minute on a DGX Spark once the 36.5 GB base image is pulled from nvcr.io (once).
# Usage: scripts/build-image.sh            IMAGE=<tag> JOBS=<n> scripts/build-image.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-tensorfold-zig:runtime-pt2607}"
JOBS="${JOBS:-8}"
[[ "$(uname -m)" == aarch64 ]] || echo "[build-image] warning: the image and its kernel set are for aarch64 GB10 (sm_121)" >&2
DOCKER_BUILDKIT=1 docker build --build-arg "JOBS=$JOBS" -t "$IMAGE" "$HERE/image"
docker run --rm --entrypoint bash "$IMAGE" -c \
  'test -s "$TENSORFOLD_CUDA_KERNELS/aot.json" && ls "$TENSORFOLD_CUDA_KERNELS/cubins" | wc -l | xargs echo "[build-image] cubins:" \
   && PYTHONPATH=$TENSORFOLD_VISION_PYTHONPATH python3 -c "import tensorfold.vision.native_helper" \
   && ldconfig -p | grep -c -E "libcudart|libcublas|libnvrtc" | xargs echo "[build-image] CUDA runtime libs:"'
echo "[build-image] -> $IMAGE"
