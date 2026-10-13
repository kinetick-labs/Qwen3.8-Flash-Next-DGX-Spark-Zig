#!/usr/bin/env bash
# Serve Qwen3.8 Flash-Next EXL3 4.05bpw with the TensorFold Zig engine on one DGX Spark (GB10).
# Foreground; Ctrl-C stops it. Settings: .env (see .env.example), overridden by the caller's environment.
set -euo pipefail
cd "$(dirname "$0")/.."
VARS=(PROFILE TF_DIR MODEL TENSORFOLD_CUDA_KERNELS HOST PORT NAME CONTEXT PARALLEL KV_DTYPE EXTRA_ARGS
      TENSORFOLD_MEMORY_RESERVE_GIB
      TF_FLASHNEXT_PREFILL_ROWS TF_FLASHNEXT_X3_MOE_WINDOW TF_FLASHNEXT_DEPTH TF_FLASHNEXT_CONFIDENCE
      TF_FLASHNEXT_PRODUCT_STREAMS TF_FLASHNEXT_TELEMETRY)
caller=(); for v in "${VARS[@]}"; do [[ -n "${!v:-}" ]] && caller+=("$v=${!v}"); done
[[ -f .env ]] && { set -a; . ./.env; set +a; }
for e in ${caller[@]+"${caller[@]}"}; do export "$e"; done

TF_DIR="${TF_DIR:-$HOME/TensorFold}"
BIN="$TF_DIR/zig-out/native/bin/tensorfold-native"
MODEL="${MODEL:-$HOME/models/exl3-405}"
HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-8083}"; NAME="${NAME:-qwen3.8-flash-next}"
# Profiles (README "Profiles"): single = 1 stream, bf16 KV, draft depth 4 (exact, the published shas);
# multi = 4 streams at 262k, FP8 KV (needs the kv8 kernels, scripts/add-kv8.sh), draft depth 3.
# Explicit variables still win over the profile.
PROFILE="${PROFILE:-single}"
case "$PROFILE" in
  single) d_par=1; d_kv=bf16; d_depth=4 ;;
  multi)  d_par=4; d_kv=fp8;  d_depth=3 ;;
  *) echo "[serve] PROFILE=$PROFILE: use single or multi" >&2; exit 2 ;;
esac
CONTEXT="${CONTEXT:-262144}"; PARALLEL="${PARALLEL:-$d_par}"; KV_DTYPE="${KV_DTYPE:-$d_kv}"
export TF_FLASHNEXT_DEPTH="${TF_FLASHNEXT_DEPTH:-$d_depth}"
export TF_FLASHNEXT_PREFILL_ROWS="${TF_FLASHNEXT_PREFILL_ROWS:-4096}"
for v in TF_FLASHNEXT_CONFIDENCE TF_FLASHNEXT_PRODUCT_STREAMS TF_FLASHNEXT_X3_MOE_WINDOW TF_FLASHNEXT_TELEMETRY TENSORFOLD_MEMORY_RESERVE_GIB; do
  [[ -n "${!v:-}" ]] && export "$v"
done
export TENSORFOLD_NO_UPDATE_CHECK=1

[[ -x "$BIN" ]] || { echo "[serve] $BIN missing: run scripts/build.sh" >&2; exit 2; }
[[ -f "$MODEL/config.json" ]] || { echo "[serve] no checkpoint at $MODEL" >&2; exit 2; }
: "${TENSORFOLD_CUDA_KERNELS:?set TENSORFOLD_CUDA_KERNELS to the sm121 Triton set (aot.json + cubins/)}"
grep -q _ple_rows "$TENSORFOLD_CUDA_KERNELS/aot.json" \
  || { echo "[serve] $TENSORFOLD_CUDA_KERNELS/aot.json has no _ple_rows: this pack cannot load" >&2; exit 2; }
if [[ "$KV_DTYPE" == fp8 ]] && ! grep -q _attn_prep8 "$TENSORFOLD_CUDA_KERNELS/aot.json"; then
  echo "[serve] KV_DTYPE=fp8 needs the FP8 KV kernels: scripts/add-kv8.sh $TENSORFOLD_CUDA_KERNELS <new dir>, then point TENSORFOLD_CUDA_KERNELS at it" >&2; exit 2
fi
export TENSORFOLD_CUDA_KERNELS

args=(serve "$MODEL" --host "$HOST" --port "$PORT" --name "$NAME"
      --context "$CONTEXT" --parallel "$PARALLEL" --kv-dtype "$KV_DTYPE")
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && args+=($EXTRA_ARGS)
echo "[serve] $(cd "$TF_DIR" && git log -1 --format='%h %s' 2>/dev/null || echo '?')"
echo "[serve] http://$HOST:$PORT/v1  model=$NAME  $BIN ${args[*]}"
echo "[serve] profile=$PROFILE parallel=$PARALLEL context=$CONTEXT kv=$KV_DTYPE"
env | grep -E '^(TF_FLASHNEXT_|TENSORFOLD_MEMORY)' | sort | sed 's/^/[serve] /' || true
exec "$BIN" "${args[@]}"
