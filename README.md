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

## Latest results (2026-11-16, prod `082aee98a`, Spark GB10)

- **Decode** (single stream, temperature 0, 8 restore checks of the prod server on 2026-11-16): fib **112.6–113.6**,
  chat **47–48**, fr **49.8–51.8** tok/s. The greedy hashes stayed the same every time (fib `c93c652df0a0`, chat
  `49127283b123`, fr `4de58665d0fb`), TC2 passed 7/7, and `/health` answered 200 about 96–99 s after a cold start.
- **Prefill:** 14k tokens solo at **2,291 tok/s** (deploy gate of this pin). No newer prefill measurement.
- **Agent jobs on prod** (Mercury bots, same server):

  | job | wall time | tokens in / out | steps / tool calls |
  |---|---|---|---|
  | `veille-colombages` (weekly property watch, vision on listing photos), 2026-11-13 | 65 min | 3.08M / 79k | 67 / 78 |
  | `veille-dgx` (daily DGX Spark offer watch), 2026-11-14 | 125 s | 52.7k / 1.8k | 7 / — |
  | `veille-dgx`, 2026-11-15 | 127 s | 37k / 1.9k | 5 / 4 |

### Concurrency (c1–c4)

Only measured numbers are shown; "not measured" means no run exists for that cell.

| streams | prefill | decode, this pin |
|---|---|---|
| c1 | 2,291 tok/s (14k solo) | 112.6–113.6 fib / 47–48 chat / 49.8–51.8 fr tok/s |
| c2 | not measured | not measured |
| c3 | not measured alone | 3-agent session (`ma.py`, 3 agents x 2 turns, prompts filling beside decode): avg 33.5, min 26.7 tok/s per stream |
| c4 | not measured | not measured (prod serves 3 lanes) |

An older c1–c4 decode run, on **2026-10-17**, used an earlier FP4-Local build, not this pin: 4 lanes, drafts on,
before the DFlash2 drafter and srf scheduling. It ran 256 tokens a stream at temperature 0, and every output was
equal to the same request run alone.

| streams | code: total / per stream tok/s | chat: total / per stream tok/s | TTFT max |
|---|---|---|---|
| c1 | 67.1 / 67.2 | 52.7 / 52.8 | 0.09 / 0.13 s |
| c2 | 120.8 / 61.1 | 93.6 / 47.3 | 0.19 / 0.26 s |
| c3 | 135.7 / 46.3 | 111.1 / 37.8 | 0.28 / 0.39 s |
| c4 | 204.0 / 52.5 | 164.9 / 42.5 | 0.37 / 0.52 s |

## 1. Checkpoint and drafter

```bash
huggingface-cli download nvidia/Qwen3.8-Flash-Next-FP4-Local --revision b9ce18a6887e18c089fdea8509a33e63429a3300
# DFlash2 drafter in ~/models/qwen38-dflash2, bf16 vision tower in ~/models/sxuff-vision-bf16/vision_tower_bf16.safetensors
```

## 2. Runtime image

serve.sh runs `tensorfold-native` (built in step 3, mounted from `$ROOT`) inside a local runtime image. Two scripts
build it, no GPU needed (the Qwen server can keep running):

```bash
scripts/build-image.sh              # -> tensorfold-zig:runtime-pt2607         (image/Dockerfile)
scripts/runner-image/build.sh       # -> tensorfold-zig:runtime-pt2607-runner  (the tag serve.sh runs)
```

`scripts/build-image.sh` (`image/Dockerfile`, multi-stage):

- **Base:** `nvcr.io/nvidia/pytorch:26.07-py3`, pinned by digest (CUDA 13 libraries, torch, Triton 3.7.1, Python
  3.12). The 36.5 GB base is pulled once; after that the build takes under a minute on a Spark.
- **Vision helper package:** `--vision-helper` runs `python3 -m tensorfold.vision.native_helper`. The image carries
  the Python package at `/opt/tensorfold/python`: upstream ashhart/TensorFold `4af95036e` (the last tree with the
  Python engine, before 1.0.0) plus `image/python-overlay.patch` (the helper, the FP8 KV kernels and the Flash Next
  kernel sources), with transformers 5.17.0 and PyAV 19.0.1.
- **Kernel set:** the mia engine replays a captured Triton kernel set from `TENSORFOLD_CUDA_KERNELS`
  (`/opt/tensorfold/share/tensorfold/cuda/sm121`: `aot.json` + 320 cubins).
  - The build compiles it for sm_121 without a GPU, using `tools/zig/flashnext_aot.py` and the spec in
    TensorFold `082aee98a`.
  - It then checks every cubin's sha256 against `image/sm121-kernels.txt`.
  - `image/python-mtimes.txt` restores the source mtimes that Triton writes into the line tables, so the cubins come
    out byte-identical.
- **Runner layer:** `scripts/runner-image/build.sh` adds clang-18/lld-18 and Jay Leaton's `runner-assets-20261107`
  (`calc.tfpc`, `graph-v7-powmod` bank). His `scripts/fetch_runner_assets.py` downloads them and checks them file by
  file against his pins (`config/runner-assets.*` in jayleaton/qwen4-tensorfold-spark).

serve.sh refuses to start without the image; `IMAGE=` selects another tag.

Checked on our Spark (2026-11-16) against the image prod runs (`tensorfold-zig:exl3-572414d-runner`, built 2026-10-09):

- **Identical:**
  - the kernel set: `aot.json` and all 320 cubins, byte for byte;
  - `pip freeze`;
  - the libraries `tensorfold-native` resolves (`ldd`, nothing missing);
  - the runner assets and clang/lld 18.1.3.
- **Differences:** one comment line in `families/qwen4_exp/cuda/kv8.py`, same length, so the cubins do not change.
  The new runner layer also ships the Qwenputer MIT notice.

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
- `image/python-overlay.patch` keeps the attributions in its files: the vision helper's request limits follow
  MiaAI-Lab's single-Spark recipe (patches 0008 and 0009), and the FP8 KV rows follow their row format.
- **Kinetick-labs** (tournierjc): DFlash2 integration, the compute lane's Flash-Next wiring, tool-less default and MPO fix, srf and cache-resume prefill scheduling, adaptive busy chunks,
  fp8 KV prod config, the runtime image build and this recipe.

Previous recipe (EXL3 4.05 bpw on the Zig engine): commit `434f376` of this repository.

TensorFold is Apache-2.0; weights are under their own licenses.
