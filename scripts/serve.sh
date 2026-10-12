#!/usr/bin/env bash
# Serve Qwen3.8 Flash-Next EXL3 4.05bpw with the TensorFold Zig engine on one DGX Spark (GB10).
# Foreground; Ctrl-C stops it. Settings: .env (see .env.example), overridden by the caller's environment.
set -euo pipefail
cd "$(dirname "$0")/.."
VARS=(TF_DIR MODEL TENSORFOLD_CUDA_KERNELS HOST PORT NAME CONTEXT PARALLEL KV_DTYPE EXTRA_ARGS
      TF_FLASHNEXT_PREFILL_ROWS TF_FLASHNEXT_X3_MOE_WINDOW TF_FLASHNEXT_DEPTH TF_FLASHNEXT_CONFIDENCE
      TF_FLASHNEXT_PRODUCT_STREAMS TF_FLASHNEXT_TELEMETRY)
caller=(); for v in "${VARS[@]}"; do [[ -n "${!v:-}" ]] && caller+=("$v=${!v}"); done
[[ -f .env ]] && { set -a; . ./.env; set +a; }
for e in ${caller[@]+"${caller[@]}"}; do export "$e"; done

TF_DIR="${TF_DIR:-$HOME/TensorFold}"
BIN="$TF_DIR/zig-out/native/bin/tensorfold-native"
MODEL="${MODEL:-$HOME/models/exl3-405}"
HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-8083}"; NAME="${NAME:-qwen3.8-flash-next}"
CONTEXT="${CONTEXT:-262144}"; PARALLEL="${PARALLEL:-2}"; KV_DTYPE="${KV_DTYPE:-bf16}"
export TF_FLASHNEXT_PREFILL_ROWS="${TF_FLASHNEXT_PREFILL_ROWS:-4096}"
export TENSORFOLD_NO_UPDATE_CHECK=1

[[ -x "$BIN" ]] || { echo "[serve] $BIN missing: run scripts/build.sh" >&2; exit 2; }
[[ -f "$MODEL/config.json" ]] || { echo "[serve] no checkpoint at $MODEL" >&2; exit 2; }
: "${TENSORFOLD_CUDA_KERNELS:?set TENSORFOLD_CUDA_KERNELS to the sm121 Triton set (aot.json + cubins/)}"
grep -q _ple_rows "$TENSORFOLD_CUDA_KERNELS/aot.json" \
  || { echo "[serve] $TENSORFOLD_CUDA_KERNELS/aot.json has no _ple_rows: this pack cannot load" >&2; exit 2; }
export TENSORFOLD_CUDA_KERNELS

args=(serve "$MODEL" --host "$HOST" --port "$PORT" --name "$NAME"
      --context "$CONTEXT" --parallel "$PARALLEL" --kv-dtype "$KV_DTYPE")
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && args+=($EXTRA_ARGS)
echo "[serve] $(cd "$TF_DIR" && git log -1 --format='%h %s' 2>/dev/null || echo '?')"
echo "[serve] http://$HOST:$PORT/v1  model=$NAME  $BIN ${args[*]}"
env | grep -E '^TF_FLASHNEXT_' | sort | sed 's/^/[serve] /' || true
exec "$BIN" "${args[@]}"
