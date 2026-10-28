#!/usr/bin/env bash
# Make an FP8-KV-capable Triton kernel set: copy SRC (an sm121 set: aot.json + cubins/) to DST and add the
# FP8 KV cache kernels (_attn_prep8, _chunks8, TP=1) shipped in kernels/kv8/. Without them `--kv-dtype fp8`
# refuses to start with "no captured Triton variant of _attn_prep8".
#   scripts/add-kv8.sh <src sm121 dir> <dst dir>      e.g. scripts/add-kv8.sh ~/tf-kernels/sm121 ~/tf-kernels-kv8/sm121
set -euo pipefail
SRC="${1:?source sm121 kernel dir}"; DST="${2:?destination dir}"
KV8="$(cd "$(dirname "$0")/../kernels/kv8" && pwd)"
[[ -f "$SRC/aot.json" && -d "$SRC/cubins" ]] || { echo "[kv8] $SRC is not a kernel set (aot.json + cubins/)" >&2; exit 2; }
[[ "$(cd "$SRC" && pwd)" != "$(mkdir -p "$DST" && cd "$DST" && pwd)" ]] || { echo "[kv8] DST must differ from SRC" >&2; exit 2; }
cp -r "$SRC/." "$DST/"
python3 - "$KV8" "$DST" <<'PY'
import hashlib, json, shutil, sys
kv8, dst = sys.argv[1], sys.argv[2]
add = json.load(open(f"{kv8}/kv8.json"))["kernels"]
path = f"{dst}/aot.json"; aot = json.load(open(path))
if aot.get("target") != ["cuda", 121, 32]:
    sys.exit(f"[kv8] {path}: target {aot.get('target')} is not sm_121")
have = {k["hash"] for k in aot["kernels"]}
n = 0
for k in add:
    cub = f"{kv8}/cubins/{k['hash']}.cubin"
    if hashlib.sha256(open(cub, "rb").read()).hexdigest() != k["cubin_sha256"]:
        sys.exit(f"[kv8] {cub}: sha256 mismatch")
    if k["hash"] in have:
        continue
    shutil.copy(cub, f"{dst}/cubins/")
    aot["kernels"].append(k); n += 1
json.dump(aot, open(path, "w"), indent=1)
print(f"[kv8] {dst}: added {n} kernels ({len(aot['kernels'])} total)")
PY
