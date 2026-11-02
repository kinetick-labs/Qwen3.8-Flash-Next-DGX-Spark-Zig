# Qwen3.8 Flash-Next FP4-Local on one DGX Spark — TensorFold (kinetick-labs main)

Serves **Qwen3.8 Flash-Next FP4-Local** (`nvidia/Qwen3.8-Flash-Next-FP4-Local`, rev `b9ce18a6`) on **one NVIDIA GB10**
with the native Zig engine (`tensorfold-native`, mia engine) of
[kinetick-labs/TensorFold](https://github.com/kinetick-labs/TensorFold) **`main` @ `80071f40c`**: upstream
[ashhart/TensorFold](https://github.com/ashhart/TensorFold) main `45fcab6c2` plus our engine work. `scripts/build.sh` pins it.

This is the configuration running in Kinetick-labs prod (port 8083).

## Configuration

| | |
|---|---|
| Weights | `nvidia/Qwen3.8-Flash-Next-FP4-Local` rev `b9ce18a6887e18c089fdea8509a33e63429a3300` |
| Streams / context | `--parallel 3` x `--context 262144` |
| KV cache | fp8 (`--kv-cache fp8`) |
| Prompt cache | `--prompt-cache-gib 12` |
| Vision | `--vision-helper`, bf16 tower (`TENSORFOLD_VISION_WEIGHTS`) |
| Drafts | DFlash2 drafter (`~/models/qwen38-dflash2`): 7 drafts, cut 0.15, wide cut 0.5 |
| Structured output | strict tools, `tool_choice: required`, `json_schema` through `libtfgrammar` |
| Prefill beside decode | `TF_FLASHNEXT_FILL=1` (layer-sliced), `TENSORFOLD_PREFILL_POLICY=srf`, `TF_FLASHNEXT_FILL_LAYERS_BUSY=1`, `TF_FLASHNEXT_FILL_ROWS_BUSY=1024`, `TF_FLASHNEXT_FILL_ROWS_HOLD_MS` 3000 (default) |
| Engine | `TENSORFOLD_FLASHNEXT_ENGINE=mia`, `TF_FLASHNEXT_PREFILL_ROWS=4096`, depth 15, confidence −0.4, product streams 2 |

Why these prefill knobs: with three agents, a long fresh prompt used to stall the other streams' decode and make
cache-resumed turns wait behind it. `srf` fills one long prompt at a time (the begun one first, else the fewest rows
left, counted from the cache match), short or resumed prompts start between its chunks, and while another stream
decodes the long prompt advances one layer and 1024 rows at a time. Result: decode min 31 tok/s (was ~5), later turns
under 1.4 s, solo prefill unchanged (bench/results.md).

## 1. Checkpoint and drafter

```bash
huggingface-cli download nvidia/Qwen3.8-Flash-Next-FP4-Local --revision b9ce18a6887e18c089fdea8509a33e63429a3300
# DFlash2 drafter in ~/models/qwen38-dflash2, bf16 vision tower in ~/models/sxuff-vision-bf16/vision_tower_bf16.safetensors
```

## 2. Build and install

```bash
TF_DIR=$HOME/TensorFold ROOT=$HOME/tf-prod-df2g LIBS_FROM=/path/to/libs scripts/build.sh
```

Builds `main` @ `80071f40c` ReleaseFast (Zig 0.17, CUDA nvcc, sm_121) and installs `bin/tensorfold-native`, `libs/`
and `COMMIT` under `$ROOT`. **Grammar library:** strict tools and `json_schema` need `libtfgrammar.so` (Linux aarch64)
in `$ROOT/libs`: take it from a TensorFold release (pinned grammar runtime) or build it with
`zig build grammar -Dxgrammar=DIR`; `libturbojpeg` / `libwebpdecoder` there serve image input.

## 3. Serve

```bash
cp scripts/serve.sh $ROOT/serve.sh && $ROOT/serve.sh        # docker container qwen38-fp4local on :8083
DRY_RUN=1 $ROOT/serve.sh                                    # print the command only
```

As a user unit: `scripts/zz-df2g.conf` is a systemd drop-in for `qwen38-flash-next-tensorfold.service`
(`ExecStart` runs serve.sh, then follows the container's logs). Rollback: remove the drop-in, `daemon-reload`, restart.

Checks after start: `/health` 200, a text and an image chat, `tool_choice: required`, `response_format: json_schema`.

## Credits

- **TensorFold**: ashhart and contributors (upstream ashhart/TensorFold, Apache-2.0).
- **MiaAI-Lab**: the mia engine for Flash Next and the layer-sliced prefill (`TF_FLASHNEXT_FILL`) this recipe tunes
  (MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold). Their recipe's tuning set is not used here.
- **Jay Leaton** (jayleaton): upstream strict-tool / grammar work our grammar hooks follow.
- **Kinetick-labs** (tournierjc): DFlash2 integration, srf and cache-resume prefill scheduling, adaptive busy chunks,
  fp8 KV prod config and this recipe.

Previous recipe (EXL3 4.05 bpw on the Zig engine): commit `434f376` of this repository.

TensorFold is Apache-2.0; weights are under their own licenses.
