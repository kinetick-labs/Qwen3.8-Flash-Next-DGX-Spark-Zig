#!/usr/bin/env bash
# The --vision helper for tensorfold-native, in a container: the Zig server runs
#   $TENSORFOLD_VISION_PYTHON -m tensorfold.vision.native_helper --model <dir> ...
# and talks to it over stdin/stdout (length-prefixed JSON frames). serve.sh sets TENSORFOLD_VISION_PYTHON to
# this script when VISION=1. The helper (TensorFold's Python tree + torch) is not in the public fork, so it comes from
# an image that ships it at /opt/tensorfold/python (VISION_IMAGE).
set -euo pipefail
IMAGE="${VISION_IMAGE:-tensorfold-zig:exl3-572414d}"
MODEL="${MODEL:?MODEL (the checkpoint dir, mounted read-only at the same path)}"
# One helper container per server port: a helper left over by a server that died is removed first.
NAME="tf-vision-helper-${PORT:-8083}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run --rm -i --init --name "$NAME" --gpus all --ipc host --network none \
  -v "$MODEL:$MODEL:ro" -e PYTHONPATH=/opt/tensorfold/python \
  -e TENSORFOLD_VISION_WEIGHTS="${TENSORFOLD_VISION_WEIGHTS:-$MODEL/vision-f16.safetensors}" \
  -e TENSORFOLD_VISION_WORKSPACE_MIB="${TENSORFOLD_VISION_WORKSPACE_MIB:-1280}" \
  --entrypoint python3 "$IMAGE" "$@"
