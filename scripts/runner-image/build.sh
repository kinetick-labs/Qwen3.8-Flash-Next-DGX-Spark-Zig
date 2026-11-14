#!/usr/bin/env bash
# Build tensorfold-zig:exl3-572414d-runner (the image serve.sh runs): the runtime image + the runner layer.
# Assets: Jay Leaton's release runner-assets-20261107, fetched and checked file by file by his
# scripts/fetch_runner_assets.py (jayleaton/qwen4-tensorfold-spark, Apache-2.0).
set -euo pipefail
BASE_IMAGE="${BASE_IMAGE:-tensorfold-zig:exl3-572414d}"
TAG="${RUNNER_IMAGE:-${BASE_IMAGE}-runner}"
JL_DIR="${JL_DIR:-$HOME/qwen4-tensorfold-spark}"
HERE="$(cd "$(dirname "$0")" && pwd)"
[[ -d "$JL_DIR/.git" ]] || git clone https://github.com/jayleaton/qwen4-tensorfold-spark "$JL_DIR"
ctx="$(mktemp -d)"; trap 'rm -rf "$ctx"' EXIT
(cd "$JL_DIR" && python3 -I scripts/fetch_runner_assets.py config/runner-assets.release config/runner-assets.sha256 "$ctx/runner")
cp "$HERE/Dockerfile" "$ctx/Dockerfile"
docker build --build-arg "BASE_IMAGE=$BASE_IMAGE" -t "$TAG" "$ctx"
echo "[runner-image] $TAG"
