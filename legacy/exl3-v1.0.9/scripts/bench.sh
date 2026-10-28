#!/usr/bin/env bash
# Single- and multi-stream decode bench against a running server (tools from the TensorFold checkout).
set -euo pipefail
TF_DIR="${TF_DIR:-$HOME/TensorFold}"; BASE="${BASE:-http://127.0.0.1:${PORT:-8083}}"; NAME="${NAME:-qwen3.8-flash-next}"
LEVELS="${LEVELS:-1}"
python3 "$TF_DIR/tools/bench_openai.py" "$BASE" "$NAME" --tokens 256 --reps 5 --temperatures 0
python3 "$TF_DIR/tools/bench_concurrent.py" "$BASE" "$NAME" --levels "$LEVELS" --tokens 256 --reps 3 \
  --temperatures 0 --alone
