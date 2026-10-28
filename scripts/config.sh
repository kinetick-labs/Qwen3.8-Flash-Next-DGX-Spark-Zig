# Shared settings for start.sh, stop.sh and scripts/*.sh. A setting's value comes from the first of these that sets it:
#   1. the environment: `PORT=9000 ./start.sh`, `PULL=0 scripts/prepare.sh`
#   2. ./.env (KEY=value lines, read, never run)
# Settings marked PROVISIONAL follow TensorFold's Zig engine (tensorfold-native) and change when a gated commit lands.
if [[ -f .env ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < .env
fi
unset _line _key _value

# Checkpoint. The default is NVIDIA's quantization-aware-distilled FP4 (NVFP4 routed experts, MXFP8 and bf16 elsewhere,
# FP8 n-gram table, MTP head): nvidia/Qwen3.8-Flash-Next-FP4-Local, ~96 GiB. azampatti's INT4-AutoRound (top-5 routing,
# GPTQ int4 experts, ~122 GiB) stays available: CHECKPOINT=int4 ./start.sh, or set MODEL_ID and MODEL_REVISION yourself.
# The engine reads the format from the checkpoint. WEIGHTS_GIB is what loading takes of MemAvailable (measured on the Zig engine:
# 81 GiB for the NVIDIA checkpoint with vision on).
CHECKPOINT="${CHECKPOINT:-nvfp4}"
_pin=1; [[ -z "${MODEL_ID:-}" ]] || _pin=0       # a MODEL_ID set by hand takes no preset revision pin
case "$CHECKPOINT" in
  nvfp4)
    : "${MODEL_ID:=nvidia/Qwen3.8-Flash-Next-FP4-Local}"
    (( ! _pin )) || : "${MODEL_REVISION:=b9ce18a6887e18c089fdea8509a33e63429a3300}"
    CKPT_GIB=96; WEIGHTS_GIB="${WEIGHTS_GIB:-81}"
    QUANT_LABEL="NVIDIA FP4 QAD: NVFP4 experts, MXFP8" ;;
  int4)
    : "${MODEL_ID:=azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound}"
    (( ! _pin )) || : "${MODEL_REVISION:=1464274120d36a4d8fcaa934552334a7d83ce0fd}"
    CKPT_GIB=122; WEIGHTS_GIB="${WEIGHTS_GIB:-66}"
    QUANT_LABEL="INT4-AutoRound" ;;
  *) echo "CHECKPOINT is nvfp4 or int4, not $CHECKPOINT" >&2; exit 1 ;;
esac
MODEL_REVISION="${MODEL_REVISION:-}"
unset _pin

# The engine: TensorFold's Zig engine (tensorfold-native), built from TF_REPO at TF_REF with patches/*.patch applied
# (git apply in the checkout's root), with Zig ZIG_VERSION. Only the Zig build goes into the image, plus the two
# image decoders the engine wants (libjpeg-turbo for JPEG, libwebp's decoder for WebP), built from pinned sources.
#
# This fork pins kinetick-labs/TensorFold's integration/1.1.8-vision: TensorFold v1.1.8 (03d08f1ef) plus the two lossless
# vision patches from MiaAI-Lab's recipe (the NVIDIA FP4 checkpoint's quantized tower decoded to bf16 at load, and the
# per-request image-token cap at 16384). Its tree is byte-identical to a v1.1.8 checkout after `git apply` of those two
# (both hash 80004eceb6), so patches/ here is deliberately empty - the pin carries them, and replaying them would be a
# zero-delta overlay. The other fifteen patches of that series are not carried; docs/audit-upstream-rebase.md says why in
# one line each (upstream carries the prompt path and the n-gram reader from 1.1.5/1.1.7, and the rest needs a port).
# v1.1.8 also loads the NVFP4 experts on several threads (#1368): about 83 s to serve instead of about 200 s.
TF_REPO="${TF_REPO:-https://github.com/kinetick-labs/TensorFold.git}"
TF_REF="${TF_REF:-ba0f4941a35719a2824150dc31d35a2e3f673058}"   # v1.1.8 + MiaAI-Lab's vision patches 0001, 0002
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
LIBJPEG_TURBO_VERSION=3.1.4.1
LIBJPEG_TURBO_SHA256=ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022
LIBWEBP_VERSION=1.6.0
LIBWEBP_SHA256=e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564
# libtfgrammar.so (structured output, tool_choice "required" and named functions) comes prebuilt from TensorFold's own release archive, checked against its SHA-256;
# the archive ships it with xgrammar's Apache-2.0 notices. TF_RELEASE is the release that matches TF_REF.
TF_RELEASE_VERSION=1.1.8
TF_RELEASE_AARCH64_SHA256=1afb157524d63a10e3dd144ec95714dcb0ed7a1e943dae247232bdf79231aad6
TF_TAG="v1.1.8-${TF_REF:0:7}"
IMAGE="${IMAGE:-tensorfold-qwen38:${TF_TAG}}"
image_hash() {
  ( cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
    cat patches/*.patch 2>/dev/null
    sed -n "/<<'DOCKERFILE'/,/^DOCKERFILE\$/p" scripts/prepare.sh
    echo "$TF_REPO $TF_REF zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE jpeg-$LIBJPEG_TURBO_VERSION-$LIBJPEG_TURBO_SHA256 webp-$LIBWEBP_VERSION-$LIBWEBP_SHA256 grammar-$TF_RELEASE_VERSION-$TF_RELEASE_AARCH64_SHA256"
  ) | sha256sum | cut -c1-12
}
# No prebuilt image for this fork: TF_REPO/TF_REF enter image_hash(), so the tag under MiaAI-Lab's GHCR name never
# matches and prepare.sh falls back to a local build (it warns "could not pull"). Set GHCR_IMAGE to a registry of our
# own once we publish one.
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold}"
IMAGE_TAG="${IMAGE_TAG:-}"
IMAGE_DIGEST="${IMAGE_DIGEST:-}"
prebuilt_image() {
  local tag="${TF_TAG}-$(image_hash)"
  if [[ "$tag" == "$IMAGE_TAG" && -n "$IMAGE_DIGEST" ]]; then echo "$GHCR_IMAGE@$IMAGE_DIGEST"; else echo "$GHCR_IMAGE:$tag"; fi
}
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-tf}"

SERVED_NAME="${SERVED_NAME:-Qwen3.8-Flash-Next}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
# Native window 262,144; the Flash Next CUDA engine of TensorFold 1.1.8 has no YaRN extension, so this is also the maximum.
CONTEXT="${CONTEXT:-262144}"
# Requests decoded together. The engine admits each one inside its own budget (MemAvailable after load, less
# TENSORFOLD_MEMORY_RESERVE_GIB, and with VISION=1 less another 2 GiB kept for the vision helper) and refuses one
# that would not fit. The pool is shared: streams grow on demand, so PARALLEL is a cap, not a promise that every
# stream can sit at a full 262,144-token window at once. On a 121 GiB Spark (NVIDIA FP4, vision on) the engine reports
# about a 24.3 GiB sequence pool after load; a full 262,144-token window is 7.5 GiB with a bf16 cache (the default here)
# and 4.4 GiB with fp8, so 3 streams fit at the full window with bf16 and 5 need KV_DTYPE=fp8 -- which is the lossy
# cache this fork does not default to. 3 is the default for that reason; raise PARALLEL with KV_DTYPE=fp8, and read
# the engine's own startup line ("up to N streams, each growing to C ... R GiB free") rather than trusting this sum.
PARALLEL="${PARALLEL:-3}"
# MTP: up to this many drafts a round (the engine's default is 6; drafts are checked, so replies match serial decoding).
export TF_FLASHNEXT_DRAFTS="${TF_FLASHNEXT_DRAFTS:-6}"
# The prompt path (fused hyper-connections, the NVFP4 prompt expert tile, the 128x256 prompt GEMM, 3072-row chunks) and the n-gram
# reader (pread with read-ahead, its mode through TF_FLASHNEXT_PLE_IO, read-ahead through TF_FLASHNEXT_WARM_AHEAD) are **in the
# engine** from TensorFold 1.1.5 and 1.1.7, on by default, with no recipe-side switch. This fork used to export MiaAI-Lab's
# patch knobs here (PROMPT_HC2, PROMPT_GEMM2, PROMPT_EXPERT_TILE_ROWS, HEAD_PIECE, GATHER_READ, DRAFT_AHEAD, KEEP_LATE, SAMPLED_HEADS,
# SAMPLE_CAND, GATHER_EARLY, HOST_DRAW_THREADS, SHARED_SIDE, GDN_LANES, PREFILL_TAIL): the pinned engine reads none of them, so they
# are gone rather than inert. docs/audit-upstream-rebase.md maps each of MiaAI-Lab's seventeen patches to what upstream now does.
# The draft chain's stop rule is the engine's own in v1.1.8 (a per-draft confidence floor, 0.6 by default). MiaAI-Lab's patch 0017
# replaces it with the db281878 running-product rule (-0.4, up to 5 drafting streams) for a few percent of greedy code decode; it
# changes how many draft rows a window asks for and never a reply, and it is not carried here (no port onto 1.1.8 yet).
export TF_FLASHNEXT_CONFIDENCE="${TF_FLASHNEXT_CONFIDENCE:-0.6}"
# KV cache (--kv-cache). bf16 is this fork's default because it is the exact one: an fp8 cache reads its codes widened to bf16, so
# it is the bf16 kernels' arithmetic on rows rounded to e4m3 -- ~1.84x the pool, and lossy (~98.8% top-1 agreement with bf16, so a
# free-running reply can differ). The streams it buys are priced in PARALLEL above: KV_DTYPE=fp8 is what makes 5 x 262,144 fit, and
# it works together with --vision. Upstream's other CUDA families take fp8 the same way, off by default.
KV_DTYPE="${KV_DTYPE:-bf16}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
THINKING="${THINKING:-1}"
DRAFTS="${DRAFTS:-1}"
# Sampling defaults (a request's own values win). The Zig server honours these flags.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
# Image and video. VISION=1 (default) passes --vision: rank 0 starts the Python helper (Pillow, PyAV, the 27-layer
# tower). VISION=0 is text only and returns the 2 GiB workspace to the KV budget. VISION_URLS=1 also accepts public
# https:// URLs; the default is data URLs only.
VISION="${VISION:-1}"
VISION_URLS="${VISION_URLS:-0}"
export TENSORFOLD_MAX_IMAGES="${TENSORFOLD_MAX_IMAGES:-50}"
MAX_VIDEOS="${MAX_VIDEOS:-4}"                                          # videos a request (--vision-max-videos)
export TENSORFOLD_IMAGE_TOKENS="${TENSORFOLD_IMAGE_TOKENS:-16384}"
export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"
export TENSORFOLD_VISION_WORKSPACE_MIB="${TENSORFOLD_VISION_WORKSPACE_MIB:-2048}"
# The engine keeps this much MemAvailable free when it sizes the KV pool. prepare.sh and start.sh refuse a Spark
# that has less than this free.
# 9 GiB since patch 0014: the prompt scratch of the fused hyper-connection path takes ~0.55 GiB more of the pool, and five full 262,144-token windows
# need 23.9 GiB; with a 10 GiB reserve and the 2 GiB vision workspace the pool left 23.8 GiB. Keep nothing else big running on the Spark.
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-9}"
export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"

# There is no int8 cache and no SSD n-gram reader: the FP8 n-gram table is loaded with the weights (WEIGHTS_GIB,
# table included: 65.7 GiB measured for INT4-AutoRound, about 81 measured for the NVIDIA checkpoint: MemAvailable fell from 117.3 to 36.3 GiB at load with vision on and 8 slots, which includes the 0.84 GiB vision tower and kernels).
# Memory at start. Weights WEIGHTS_GIB. One full window's cache is kv_gib below. Prefill scratch is outside the engine's
# cache budget; 8 GiB here is slack for a 2048-row chunk, not a measured 1M-prompt peak. MEM_FLOOR_GIB is the line
# start.sh stops the server at while it loads, and the line prepare.sh refuses to start below. MEM_CHECK=0 turns
# both into a warning.
MEM_NEED_GIB="${MEM_NEED_GIB:-}"
MEM_FLOOR_GIB="${MEM_FLOOR_GIB:-$TENSORFOLD_MEMORY_RESERVE_GIB}"
# kv_gib <tokens>: one sequence at TP=1, from State.cacheBytes (cuda_state.zig).
# bf16: 30,784 bytes a token. fp8: 17,888 (keys carry both scales, values are codes, indexer and pooled stay bf16).
kv_gib() {
  local ctx=${1:-$CONTEXT} bytes=30784
  [[ "$ctx" =~ ^[0-9]+$ ]] || ctx=$CONTEXT
  [[ "$KV_DTYPE" == fp8 ]] && bytes=17888
  echo $(( (bytes * ctx + 1073741823) / 1073741824 ))
}
# mem_need_gib <context>: MemAvailable a start needs: weights, one full window, scratch slack, the vision workspace
# when VISION=1, floor, 2 GiB.
mem_need_gib() {
  local ctx=${1:-$CONTEXT} vision=0
  [[ "$VISION" == 1 ]] && vision=2
  echo $(( WEIGHTS_GIB + $(kv_gib "$ctx") + 8 + vision + MEM_FLOOR_GIB + 2 ))
}

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-qwen38}"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/qwen38-flash-next-tf}"
LOG_DIR="${LOG_DIR:-$HOME/.cache/tensorfold-qwen38/logs}"
LOG_KEEP="${LOG_KEEP:-10}"
MIN_FREE_GB="${MIN_FREE_GB:-$((CKPT_GIB + 18))}"
IMAGE_FREE_GB="${IMAGE_FREE_GB:-40}"

_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
model_cache_dir() { echo "$HF_CACHE/hub/models--${MODEL_ID//\//--}"; }
snapshot_rev() {
  if [[ -n "$MODEL_REVISION" ]]; then echo "$MODEL_REVISION"
  else cat "$(model_cache_dir)/refs/main" 2>/dev/null; fi
}

PREPARED_MARKER="$STATE_DIR/prepared"
prepared_state() {
  local hash label rev
  hash=$(image_hash)
  label=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo missing)
  rev=$(snapshot_rev)
  echo "model=$MODEL_ID@$rev image=$label patches=$hash"
}
