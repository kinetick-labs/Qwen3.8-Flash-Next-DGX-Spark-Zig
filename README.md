# Qwen3.8 Flash-Next on one DGX Spark — TensorFold Zig engine

Serve **Qwen3.8 Flash-Next** (`qwen4_exp`) from turboderp's **EXL3 4.05 bpw** pack on **one NVIDIA GB10**
(DGX Spark, 128 GB unified memory) with the native Zig engine of
[kinetick-labs/TensorFold](https://github.com/kinetick-labs/TensorFold) (`tensorfold-native`, no Python runtime).

Build from branch **`perf/zig-exl3-decode`**
([PR #14](https://github.com/kinetick-labs/TensorFold/pull/14), stacked on the prefill kernels of
[PR #13](https://github.com/kinetick-labs/TensorFold/pull/13)), rebased on upstream 1.0.8.
Measured at `6a9ae92f` (same tree as the PR #14 head `5fce75f1`). PR stack: #15 → #16 → #13 → #14.

Text only: the Zig engine does not load the vision tower.

## Summary

| | |
| --- | --- |
| Weights | `turboderp/Qwen3.8-Flash-Next-exl3`, branch `4.05bpw_h6_ng6`, rev `55a732e0c4c3d4614bc42b68493bb930d9b02c0a` |
| Engine | `tensorfold-native` built with Zig 0.17.0, CUDA fatbins for sm_121 embedded |
| Triton set | `TENSORFOLD_CUDA_KERNELS` → `sm121/` (`aot.json` + `cubins/`, must contain `_ple_rows`) |
| Context | 262,144 tokens |
| Streams | `--parallel 2` (1–4 measured below) |
| KV cache | bf16 |
| Experts window | 4096 rows (`TF_FLASHNEXT_X3_MOE_WINDOW`, default) |
| Prompt chunk | 4096 rows (`TF_FLASHNEXT_PREFILL_ROWS=4096`; engine default 2048) |
| MTP drafts | cap 4, running-product stop −0.6 (defaults on this branch) |
| Draft head | 4-bit EXL3 head over a 95,111-token draft vocabulary (EN/code + FR/ES/DE) |
| CUDA graphs | on |

## 1. Checkpoint

```bash
python3 -c "from huggingface_hub import snapshot_download as d; d('turboderp/Qwen3.8-Flash-Next-exl3', \
  revision='55a732e0c4c3d4614bc42b68493bb930d9b02c0a', local_dir='$HOME/models/exl3-405')"
```

About 101 GiB. The ~39 GB `ngram_embedding.safetensors` is memory-mapped, not loaded resident.
Tokenizer, chat template and config are read from the same folder.

## 2. Build

```bash
TF_DIR=$HOME/TensorFold TF_REF=perf/zig-exl3-decode scripts/build.sh
```

which runs, from the clone root (`build.zig` is at the root):

```bash
zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
# -> zig-out/native/bin/tensorfold-native
```

- **Fatbins.** `-Dnvcc` compiles the engine's `.cu` kernels (EXL3 `fn_exl3`, grouped experts, decode
  arm, draft head…) for sm_121 and embeds them in the binary. `-Dfatbins=DIR` embeds a prebuilt flat
  directory instead. A fatbin older than the branch will not have the new kernel symbols: rebuild.
- **Triton kernel dir.** The glue kernels (`_hc_*`, `_ple_rows`, …) are an ahead-of-time Triton
  capture, `sm121/aot.json` + `sm121/cubins/` (generator `tools/zig/flashnext_aot.py`). Point
  `TENSORFOLD_CUDA_KERNELS` at the `sm121` folder, e.g. the one shipped in a TensorFold release
  under `share/tensorfold/cuda/sm121`. Without `_ple_rows` the pack fails with
  `error.MissingTritonVariant`:
  ```bash
  grep -c _ple_rows "$TENSORFOLD_CUDA_KERNELS/aot.json"   # must be non-zero
  ```

## 3. Serve

```bash
cp .env.example .env    # edit paths
scripts/serve.sh
```

Equivalent command:

```bash
export TENSORFOLD_CUDA_KERNELS=$HOME/tf-kernels/sm121 TENSORFOLD_NO_UPDATE_CHECK=1
export TF_FLASHNEXT_PREFILL_ROWS=4096
$TF_DIR/zig-out/native/bin/tensorfold-native serve $HOME/models/exl3-405 \
  --host 127.0.0.1 --port 8083 --name qwen3.8-flash-next \
  --context 262144 --parallel 2 --kv-dtype bf16
```

OpenAI-compatible at `http://127.0.0.1:8083/v1`. The endpoint has no auth: keep it on localhost or
behind a proxy. Depth and confidence are read from the environment (`TF_FLASHNEXT_*`), not from the
`--mtp-*` server flags.

### Defaults and switches

| Variable | Default | Meaning |
| --- | --- | --- |
| `TF_FLASHNEXT_DEPTH` | 4 | Draft cap (most drafts a round may ask; the acceptance-rate rule asks inside it) |
| `TF_FLASHNEXT_CONFIDENCE` | −0.6 | Negative: running-product stop at that threshold. Positive: Python's per-draft rule (0.70 = the pack's MTP-6/0.70 profile) |
| `TF_FLASHNEXT_PRODUCT_STREAMS` | 2 | Product rule while ≤ N lanes draft, then `wide_confidence` 0.5 |
| `TF_FLASHNEXT_PREFILL_ROWS` | 2048 | Prompt chunk rows; 4096 is the measured prefill point |
| `TF_FLASHNEXT_X3_MOE_WINDOW` | 4096 | Grouped-experts window (scratch vs. passes; bits unchanged) |
| `TF_FLASHNEXT_TELEMETRY` | off | `=1`: one `tf-draft` line per round + summaries every 32 rounds; tokens unchanged |
| `TF_FLASHNEXT_GRAPH_LOG` | off | `=1`: count captured CUDA graphs |

Kill switches (A/B and bisecting; all on by default, `=0` restores the previous path with identical output):

| Variable | Turns off |
| --- | --- |
| `TF_FLASHNEXT_X3_GM` | Grouped-GEMM experts path for prompt-sized calls |
| `TF_FLASHNEXT_X3_SIDE` | Dense projections on a second stream during prefill |
| `TF_FLASHNEXT_X3_COMBINE` | Fused `down` + combine (bf16) |
| `TF_FLASHNEXT_X3_DIRECT` | Direct writes of expert outputs (no strided copy) |
| `TF_FLASHNEXT_X3_DRAFT_HEAD` | 4-bit draft head over the reduced draft vocabulary |
| `TF_FLASHNEXT_X3_ASYNC` | 3-stage cp.async ring in the 1–3-row decode expert kernel |
| `TF_FLASHNEXT_X3_ARM` | Decode expert arm (3 = default; 4 = FMA variant, slower and not bit-exact) |

## 4. Measured

One GB10 (sm_121), the serve above, temperature 0. Decode tok/s = (completion tokens − 1) /
(last token − first token), 256 tokens, median of 5 reps. Prompts are `tools/bench_openai.py`'s
(`fib` = "Write a short Python function that computes the Fibonacci sequence and explain it.").

| Metric | Value |
| --- | ---: |
| Prefill, 14,432-token prompt | **~1046 tok/s** (`token_sha afccd937e6ac`) |
| Decode, fib | **63.5 tok/s** (`b8f1f786dbd6`) |
| Decode, chat | **49.5 tok/s** (`a53cc7fa86a7`) |
| Decode, French prose | **56.4 tok/s** |
| Draft chain (cap 4 / −0.6) | ~2.24 tokens/round, acceptance by level .65 / .87 / .81 / .78 |

Starting point of this work on the same machine: prefill ~773 tok/s, fib 50.4, chat 42.7.
Every optimisation kept the token shas above unchanged.

### Multi-session (`--parallel` 1–4)

Greedy, 256 tokens, N identical requests at once (full table incl. mixed fib/chat: [bench/results.md](bench/results.md)).

| `--parallel` | Context | fib aggregate / per stream | chat aggregate / per stream | TTFT max |
| --- | ---: | ---: | ---: | ---: |
| 1 | 262,144 | 63.4 / 63.4 | 49.6 / 49.6 | 0.13 s |
| 2 | 262,144 | 110.0 / 55.5 | 88.6 / 44.4 | 0.27 s |
| 3 | 131,072 | 151.8 / 51.5 | 119.9 / 40.8 | 0.39 s |
| 4 | 98,304 | 181.4 / 46.5 | 123.2 / 31.5 | 0.53 s |

No failure, no OOM, and every concurrent reply matched the same request served alone (`token_sha`).
With bf16 KV the server refuses (it does not OOM) `--parallel 3` or `4` at 262k and `--parallel 4` at 128k:
use `--parallel 2` at 262k, `3` at 128k, `4` at 96k.

## Licenses

TensorFold is Apache-2.0. The weights are under the
[Qwen Community License](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3); review it before
downloading or deploying. This repo's scripts are MIT.
