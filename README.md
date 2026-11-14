# Qwen3.8 Flash-Next FP4-Local on one DGX Spark — TensorFold (kinetick-labs main)

Serves **Qwen3.8 Flash-Next FP4-Local** (`nvidia/Qwen3.8-Flash-Next-FP4-Local`, rev `b9ce18a6`) on **one NVIDIA GB10**
with the native Zig engine (`tensorfold-native`, mia engine) of
[kinetick-labs/TensorFold](https://github.com/kinetick-labs/TensorFold) **`main` @ `082aee98a`**: upstream
[ashhart/TensorFold](https://github.com/ashhart/TensorFold) main `d6c4cce60` (v1.2.8) plus our engine work (mid-prefill
drain fix, `TENSORFOLD_PREFILL_POLICY=srf` accepted again, DFlash2 frame-attention call fix) and **Jay Leaton's compute
lane and in-model runner** ([jayleaton/qwen4-tensorfold-spark](https://github.com/jayleaton/qwen4-tensorfold-spark),
Apache-2.0; his 33 commits keep his authorship). `scripts/build.sh` pins it.

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
| Compute lane | `TF_COMPUTE_LANE=1 TF_COMPUTE_LANE_DEFAULT=1`: thinking chats **without tools of their own** get a built-in sandboxed `python` tool (no network, no writes, 10 s, 1 GiB), run server-side and hidden from the client; `TF_COMPUTE_LANE_DEFAULT_WITH_TOOLS=1` also offers it next to a client's tools (off: agents such as Hermes render byte for byte as without the lane); `{"type":"code_interpreter"}` opts in explicitly |
| In-model runner | `TF_RUNNER=1`, reasoning spans only, `calc.tfpc` circuit + `graph-v7-powmod` bank, digit stop: exact values for expressions written while thinking (all thinking chats, with or without tools) |
| Engine | `TENSORFOLD_FLASHNEXT_ENGINE=mia`, `TF_FLASHNEXT_PREFILL_ROWS=4096`, depth 15, confidence −0.4, product streams 2 |

Compute lane, measured (19 exact math/logic + coding tasks, thinking on, no client tools, temperature 0, 2 at a time;
`bench/results.md`): **19/19** pass, **314** tokens a task (152 reasoning), **6.8 s** a task on the candidate (6.3 s on
prod `082aee98a`), against **14/19**, **1,984** tokens (1,929 reasoning, 5 cut by length) and **36.7 s** without the lane
(`17334bddd`). The lane adds to DFlash2 (19/19 at 7.3 s with drafts off). Hermes-like agent sessions (own tools) keep
the lane-off prompts and token counts (completion ~350 a task, 0 failures).

Vision: a camera's multi-picture JPEG (MPO: primary picture + embedded preview, which the helper's Pillow refused as
"multipage") is sent to the helper as its primary picture; animated PNG / WebP stay refused.

Why these prefill knobs: with three agents, a long fresh prompt used to stall the other streams' decode and make
cache-resumed turns wait behind it. `srf` fills one long prompt at a time (the begun one first, else the fewest rows
left, counted from the cache match), short or resumed prompts start between its chunks, and while another stream
decodes the long prompt advances one layer and 1024 rows at a time. Result on the previous pin: decode min 31 tok/s
(was ~5), later turns under 1.4 s, solo prefill unchanged (bench/results.md).

Measured on this pin (`082aee98a`, Spark GB10, same serve.sh as prod :8083, deploy gate against the `17334bddd`
baseline): solo prefill 14k 2,291 tok/s (baseline 2,276); decode fib 112.7 / chat 47.3 / fr 50.9 tok/s with the same
greedy decode hashes (fib `c93c652df0a0`, chat `49127283b123`, fr `4de58665d0fb`); TC2 7/7; crash repro pass.

## 1. Checkpoint and drafter

```bash
huggingface-cli download nvidia/Qwen3.8-Flash-Next-FP4-Local --revision b9ce18a6887e18c089fdea8509a33e63429a3300
# DFlash2 drafter in ~/models/qwen38-dflash2, bf16 vision tower in ~/models/sxuff-vision-bf16/vision_tower_bf16.safetensors
```

## 2. Runner image

```bash
scripts/runner-image/build.sh       # tensorfold-zig:exl3-572414d -> tensorfold-zig:exl3-572414d-runner
```

Adds clang-18/lld-18 and Jay Leaton's `runner-assets-20261107` (`calc.tfpc`, `graph-v7-powmod` bank), downloaded and
checked file by file with his `scripts/fetch_runner_assets.py` and pins (`config/runner-assets.*` in
jayleaton/qwen4-tensorfold-spark). serve.sh refuses to start without the image.

## 3. Build and install

```bash
TF_DIR=$HOME/TensorFold ROOT=$HOME/tf-prod-df2g LIBS_FROM=/path/to/libs scripts/build.sh
```

Builds `main` @ `082aee98a` ReleaseFast (Zig 0.17, CUDA nvcc, sm_121) and installs `bin/tensorfold-native`, `libs/`
and `COMMIT` under `$ROOT`. **Grammar library:** strict tools and `json_schema` need `libtfgrammar.so` (Linux aarch64)
in `$ROOT/libs`: take it from a TensorFold release (pinned grammar runtime) or build it with
`zig build grammar -Dxgrammar=DIR`; `libturbojpeg` / `libwebpdecoder` there serve image input.

## 4. Serve

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
- **Jay Leaton** (jayleaton): the compute lane (built-in sandboxed python tool), the in-model runner and digit stop,
  their runner assets and knob profile (jayleaton/qwen4-tensorfold-spark, Apache-2.0, see its NOTICE), and the
  upstream strict-tool / grammar work our grammar hooks follow.
- **Kinetick-labs** (tournierjc): DFlash2 integration, the compute lane's Flash-Next wiring, tool-less default and MPO fix, srf and cache-resume prefill scheduling, adaptive busy chunks,
  fp8 KV prod config and this recipe.

Previous recipe (EXL3 4.05 bpw on the Zig engine): commit `434f376` of this repository.

TensorFold is Apache-2.0; weights are under their own licenses.
