# Qwen3.8 Flash-Next on one DGX Spark — TensorFold Zig engine

Serve **Qwen3.8 Flash-Next** (`qwen4_exp`) from turboderp's **EXL3 4.05 bpw** pack on **one NVIDIA GB10**
(DGX Spark, 128 GB unified memory) with the native Zig engine of
[kinetick-labs/TensorFold](https://github.com/kinetick-labs/TensorFold) (`tensorfold-native`, no Python runtime).

Build from **[kinetick-labs/TensorFold](https://github.com/kinetick-labs/TensorFold) `001de3a7`**, the head of
branch `perf/zig-exl3-decode` ([PR #14](https://github.com/kinetick-labs/TensorFold/pull/14), stacked on the
prefill kernels of [PR #13](https://github.com/kinetick-labs/TensorFold/pull/13)). `scripts/build.sh` pins it.
All numbers below were measured at that commit, through this repo's `scripts/build.sh` and `scripts/serve.sh`,
on 2026-10-13.

Text only: the Zig engine does not load the vision tower.

## Summary

| | single profile | multi profile |
| --- | --- | --- |
| Weights | `turboderp/Qwen3.8-Flash-Next-exl3` `4.05bpw_h6_ng6` rev `55a732e0` | same |
| Streams / context | 1 × 262,144 | 4 × 262,144 |
| KV cache | bf16 (exact) | fp8 (needs the kv8 kernels, see below) |
| MTP drafts | cap 4, product stop −0.6 | cap 3, product stop −0.6 |
| Decode fib / chat | 62.1 / 49.7 tok/s | 178.2 / 139.9 tok/s aggregate, 45.2 / 35.7 per stream |
| Prefill, 14k prompt | 1079 tok/s | 1011 tok/s (fp8, 1 stream) |
| Prompt chunk | 4096 rows (`TF_FLASHNEXT_PREFILL_ROWS`) | same |
| Draft head | 4-bit EXL3 head over a 95,111-token draft vocabulary | same |
| CUDA graphs | on | on |

## Profiles

`PROFILE=single` (default) or `PROFILE=multi` in `.env`; explicit `PARALLEL`, `KV_DTYPE`, `TF_FLASHNEXT_DEPTH`
win over the profile. `serve.sh` also forwards `TF_FLASHNEXT_PRODUCT_STREAMS`, `TF_FLASHNEXT_CONFIDENCE` and
`TENSORFOLD_MEMORY_RESERVE_GIB`.

- **single**: one user. bf16 KV and depth 4 are the fastest per stream and reproduce the token shas below.
- **multi**: up to 4 concurrent users at the full window. fp8 KV halves the attention cache (1056 vs 2048 bytes
  a position a layer, 3.76 GiB a stream at 262k), which is what lets 4 streams fit at 262k next to a 12 GiB
  reserve. bf16 KV only fits `--parallel 2` at 262k, 3 at 128k, 4 at 96k. Depth 3 is the best
  multi-stream draft cap (table below).

### FP8 KV kernels

The sm121 Triton sets this recipe points at have no FP8 KV attention kernels, and `--kv-dtype fp8` stops at
startup with `no captured Triton variant of _attn_prep8`. This repo ships them in
[`kernels/kv8/`](kernels/kv8/README.md) (5 cubins, sha256-checked, source documented there). Make an
fp8-capable copy of your set once:

```bash
scripts/add-kv8.sh $HOME/tf-kernels/sm121 $HOME/tf-kernels-kv8/sm121
# then TENSORFOLD_CUDA_KERNELS=$HOME/tf-kernels-kv8/sm121 (bf16 serves from it too)
```

`serve.sh` refuses `KV_DTYPE=fp8` with a kernel set that lacks them.

## 1. Checkpoint

```bash
python3 -c "from huggingface_hub import snapshot_download as d; d('turboderp/Qwen3.8-Flash-Next-exl3', \
  revision='55a732e0c4c3d4614bc42b68493bb930d9b02c0a', local_dir='$HOME/models/exl3-405')"
```

About 101 GiB. The ~39 GB `ngram_embedding.safetensors` is memory-mapped, not loaded resident.
Tokenizer, chat template and config are read from the same folder.

## 2. Build

```bash
TF_DIR=$HOME/TensorFold scripts/build.sh      # checks out 001de3a7
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
export TF_FLASHNEXT_PREFILL_ROWS=4096 TF_FLASHNEXT_DEPTH=4
$TF_DIR/zig-out/native/bin/tensorfold-native serve $HOME/models/exl3-405 \
  --host 127.0.0.1 --port 8083 --name qwen3.8-flash-next \
  --context 262144 --parallel 1 --kv-dtype bf16
```

multi: `--parallel 4 --kv-dtype fp8`, `TF_FLASHNEXT_DEPTH=3`, kv8 kernel set.

OpenAI-compatible at `http://127.0.0.1:8083/v1`. The endpoint has no auth: keep it on localhost or
behind a proxy. Depth and confidence are read from the environment (`TF_FLASHNEXT_*`), not from the
`--mtp-*` server flags.

### Defaults and switches

| Variable | Default | Meaning |
| --- | --- | --- |
| `TF_FLASHNEXT_DEPTH` | 4 (multi profile: 3) | Draft cap (most drafts a round may ask; the acceptance-rate rule asks inside it) |
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

One GB10 (sm_121), `001de3a7`, temperature 0, `scripts/serve.sh`. Decode = `tools/bench_concurrent.py`
(256 tokens, median of 3 reps, N identical requests at once). Aggregate = all streams' tokens / wall span.
Prompts: `fib` = "Write a short Python function that computes the Fibonacci sequence and explain it.",
`chat` = the GPU matrix-multiplication explainer.

### Single stream (262k)

| | bf16 KV | fp8 KV |
| --- | ---: | ---: |
| Decode, fib | **62.1 tok/s** (`b8f1f786dbd6`) | 56.7 tok/s |
| Decode, chat | **49.7 tok/s** (`a53cc7fa86a7`) | 45.6 tok/s |
| Prefill, 14,432-token prompt | **1079 tok/s** (`afccd937e6ac`) | 1011 tok/s (`afccd937e6ac`) |
| Prefill, 123,519-token prompt | 586 tok/s | 577 tok/s |

### Multi-session (fib / chat, aggregate tok/s, per stream in brackets)

| `--parallel` | Context | KV | Depth | fib | chat |
| --- | ---: | --- | ---: | ---: | ---: |
| 1 | 262,144 | bf16 | 4 | 62.1 | 49.7 |
| 2 | 262,144 | bf16 | 4 | 109.3 (54.9) | 79.1 (40.4) |
| 2 | 262,144 | fp8 | 4 | 102.1 (51.0) | 82.5 (41.5) |
| 3 | 131,072 | bf16 | 4 | 151.3 (51.2) | 99.8 (33.9) |
| 3 | 262,144 | fp8 | 4 | 138.6 (46.3) | 111.3 (37.2) |
| 4 | 98,304 | bf16 | 4 | 181.1 (46.3) | 122.7 (31.9) |
| 4 | 98,304 | bf16 | 3 | 186.2 (47.8) | 148.1 (38.0) |
| 4 | 131,072 | fp8 | 4 | 168.8 (42.8) | 137.4 (35.2) |
| **4** | **262,144** | **fp8** | **3** | **178.2 (45.2)** | **139.9 (35.7)** |

No OOM, no failed request. With bf16 KV the server refuses (it does not OOM) `--parallel 3`/`4` at 262k and
`--parallel 4` at 128k. fp8 at `--parallel 4` / 262k: 4 × 3.76 GiB streams of 17.8 GiB left after a 12.2 GiB
reserve; the box kept ≥ 29 GiB available during the bench.

### Draft cap and `PRODUCT_STREAMS` with several streams (bf16, aggregate fib / chat)

| | p3 / 128k | p4 / 96k |
| --- | ---: | ---: |
| depth 4, PS 2 (default) | 151.3 / 99.8 | 181.1 / 122.7 |
| depth 4, PS 1 | 151.5 / 101.2 | 181.4 / 145.2 |
| depth 4, PS 3 | 153.6 / 118.3 | 181.0 / 123.9 |
| depth 3, PS 2 | 152.0 / 120.1 | **186.2 / 148.1** |
| depth 3, PS 1 | 150.9 / 99.9 | — |
| depth 2, PS 2 | 144.2 / 120.5 | 178.5 / 149.7 |
| depth 2, PS 1 | 144.0 / 119.9 | — |

Token shas are unchanged in every cell (drafts never change greedy output). The chat column is noisy at p3
(~100 vs ~120 between otherwise similar cells). Depth 3 is the best overall: fib keeps depth 4's speed, and chat gains
15-20% at 4 streams. `PRODUCT_STREAMS` has no consistent effect.

### fp8 KV: quality caveats

- fp8 changes the tokens. fib/chat shas become `c2250bdc1427` / `65e65b64b031` (stable across 1-4 streams). The
  14k prefill's first token and a 3.9k-token summary prompt were unchanged.
- Text comparison against bf16 (400 tokens, T=0): French TCP/UDP and the GPU explainer diverge after ~80-300
  characters into different but equally fluent and correct answers. No degradation was seen.
- fp8 costs ~8% single-stream decode and ~6% prefill. Use it for capacity, not speed.
- **Needle test is inconclusive.** A passphrase hidden at 25% / 75% of a 123.5k-token prompt (TensorFold docs
  and tools) was missed by **both** bf16 and fp8. The model summarized the document, answered from it, or
  produced a wrong passphrase. The haystack (technical docs with their own instructions) is a poor needle
  test. Long-context retrieval of fp8 vs bf16 is not established here.

Starting point of this work on the same machine: prefill ~773 tok/s, fib 50.4, chat 42.7.

## Licenses

TensorFold is Apache-2.0. The weights are under the
[Qwen Community License](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3); review it before
downloading or deploying. This repo's scripts are MIT.
