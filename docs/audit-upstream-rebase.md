# The upstream rebase: what MiaAI-Lab's series still adds, and what its speed costs in quality

Supersedes the v1.1.4 pass in [`audit-v1.1.4-rebase.md`](audit-v1.1.4-rebase.md), which stays as the
record of what this repository's own recipe (the EXL3 4.05 bpw arm) had added and what a TensorFold
v1.1.4 base retires. This page is the stable one: the base moves, the verdicts below do not.

The base is **TensorFold v1.1.8**, not v1.1.4 plus someone else's seventeen patches. That is a correction,
not a preference: the prompt path MiaAI-Lab's series was written for is *in the engine* from 1.1.5 and
1.1.7, so replaying it would re-implement what upstream already ships with its own exactness gates.

## 1. What this fork pins

`kinetick-labs/TensorFold` `integration/1.1.8-vision` (`ba0f4941a35719a2824150dc31d35a2e3f673058`):
v1.1.8 (`03d08f1ef`) plus **two** patches, both from MiaAI-Lab's series and both lossless:

- `0001` — the NVIDIA FP4 checkpoint's quantized vision tower (MXFP8 attention, NVFP4 MLP and merger)
  decoded to bf16 at load, so `--vision` works on `nvidia/Qwen3.8-Flash-Next-FP4-Local`;
- `0002` — the cap on one request's image tokens at 16384 instead of 4096, so `--vision-image-tokens 16384`
  is accepted instead of refused. The default stays 4096; the recipe passes 16384 explicitly.

Both apply cleanly on every tag from v1.1.4 to v1.1.8, and neither has an upstream equivalent:
`zig/src/vision/qwen_cuda_dequant.zig` (0001's new file) does not exist at v1.1.8, and v1.1.8 still caps
a request's image rows at 4096 (`qwen_cuda_media.zig`, `max_image_tokens`). Provenance is by content: the
branch's tree hash equals a pristine v1.1.8 worktree after `git apply` of the two — `80004eceb6` on both
sides. Moving the base from 1.1.7 to 1.1.8 cost nothing: no hunk moved, the two patches applied as they
did before, and only the tree hash changed.

## 2. What 1.1.8 changes for this recipe

Nothing in the token stream, and one thing worth having:

- **Faster start.** Flash Next with NVFP4 experts now loads them on several threads straight from the
  checkpoint's mapped planes, staged through the store's pinned buffer: about 83 s instead of about 200 s
  on a GB10 with `Qwen3.8-Flash-Next-FP4-Local` (#1368). Same replies, prompt and decode unchanged or
  slightly faster. That is the difference between a two-minute restart and a one-minute one, which is what
  a reboot or a `docker stop` actually costs on this rig.
- KV pool bookkeeping is O(1) (admission no longer scans every page: ~55 µs at 262k pages before), with the
  same page order and the same replies. Conversation states can also stay on disk on CUDA
  (`--learn-conversations`, `--spill-gib`, `--checkpoint-slots`), which this recipe does not use yet.
- No new Flash Next prompt or decode work, and nothing that takes `0001`/`0002`'s place.

## 3. The other fifteen patches: absorbed, or not carried

Measured on a pristine tag worktree, each patch on its own and then the series in filename order. Only
`0001` and `0002` land; `0003` starts the conflicts on 1.1.5, 1.1.6, 1.1.7 and 1.1.8 alike, because 1.1.5
rewrote the attention cache (pages, core pool) that patch touches.

| patch | verdict on v1.1.8 | evidence |
| --- | --- | --- |
| 0001 vision FP4 tower | **carried** | new file `zig/src/vision/qwen_cuda_dequant.zig`, absent upstream |
| 0002 image token cap 16384 | **carried** | upstream still caps at 4096 (`max_image_tokens`) |
| 0003 `--kv-cache fp8` for Flash Next | not carried — **the lossy one** (see §4) and needs a port | upstream has no `kv8` in `zig/src/families/flashnext` at any tag up to v1.1.8; the patch no longer applies since 1.1.5's paged caches |
| 0004 prompt expert tile | absorbed upstream | `zig/kernels/cuda/nvfp4_experts_tile.cu` + the experts-tile module, kernel-contract entries, changelog "NVFP4 experts on prompt tiles" (1.1.5) |
| 0005 fp8 reader + 4-bit MTP cache | not carried (tied to 0003) | drafts-only either way; no value without the fp8 cache |
| 0006 prefill speed | partly absorbed | 1.1.5: no zero fill of ModelOpt's read-back, hyper-connection normalize over 256-column slices, half tiles; remainder would need a port |
| 0007 sampled requests in the batched pass | not carried | no `sampled` path in v1.1.8's `cuda_lanes_draft.zig`; port candidate |
| 0008 GPU top-k candidates | not carried | no `fn_row_cand`/`row_cand` at v1.1.8; port candidate (its candidate set is the request's `top_k`, i.e. the same set the host draw keeps) |
| 0009 round-cost model | not carried — off by default | zero value at the default, and no upstream home for the module |
| 0010 n-gram gather of all lanes in one pass | not carried | port candidate |
| 0011 host draws on worker threads | not carried | port candidate |
| 0012 DeltaNet kept-rows log | not carried | port candidate |
| 0013 n-gram pread reader + chunk tail | absorbed upstream | `zig/src/families/flashnext/cuda_ple_reader.zig` and `cuda_ple_ahead.zig` (1.1.7), driven by `TF_FLASHNEXT_PLE_IO` and `TF_FLASHNEXT_WARM_AHEAD` |
| 0014 prompt hyper-connection fusion | partly absorbed | 1.1.5: hyper-connection normalize and the dense mix as the engine does them; the fused-kernel set is upstream's now |
| 0015 MX128 two-stage prompt tile | overlapping, not equal | 1.1.7 has a persistent 128x256 tile (#1350) but the changelog calls Flash Next NVFP4 *unchanged* by it, so this one may still add something — a port and a bench, not an assumption |
| 0016 shared expert side stream + lane-batched DeltaNet | not carried | no batch-side module upstream; port candidate |
| 0017 draft running-product stop rule | not carried | v1.1.8 keeps the per-draft confidence floor; the rule changes draft rows only |

## 4. The speed/quality trade

Short version: **the only quality the recipe trades is the KV cache, and the trade is in the recipe's
defaults, not in the engine.**

- **`KV_DTYPE=fp8` is lossy by construction.** An fp8 cache stores keys and values as e4m3 codes with a
  power-of-two scale per position and KV head, and attention reads them widened back to bf16: the bf16
  kernels' arithmetic on rows rounded to e4m3. MiaAI-Lab's own README prices it — "about 1.84x the pool of
  bf16, and lossy (~98.8% top-1 agreement with a bf16 cache, so a free-running reply can differ)" — and
  their published five-stream tables at 262,144 tokens are taken with it, because 5 x 4.4 GiB fits where
  5 x 7.5 GiB does not. Their README's default *is* the lossy configuration.
- **So this fork defaults to bf16 and pays in streams, not in bits.** `KV_DTYPE=bf16`, `PARALLEL=2` at
  262,144 tokens: the engine's own admission line on this rig reads "`--parallel 3` needs 23.43 GiB for its
  streams at a 262144-token window, and the CUDA memory budget leaves 21.55 GiB" — 7.81 GiB a window, so two
  fit and three are refused before the server listens. `PARALLEL=5` is available and honest with `KV_DTYPE=fp8`
  (MiaAI-Lab's 4.4 GiB window), and `scripts/config.sh` says so in one line rather than burying it. Upstream's
  other CUDA families take `--kv-cache fp8` the same way: available, off by default.
- **The rest of their series does not touch a reply.** The 4-bit MTP head rows (0005) and the running-product
  stop rule (0017) change how many draft rows a window asks for, and drafts are verified against the target,
  so they are an *acceptance* trade, not a quality one; the prompt and gather patches (0006, 0007, 0008,
  0010, 0011, 0012, 0014, 0015, 0016) are claimed bit-identical, and 0008's candidate set is the request's
  own `top_k` — the same set the host draw keeps. None of them is carried here, so no bit-exactness question
  is open on this branch either way.
- **What is genuinely unpriced**: whether upstream's 1.1.5/1.1.7 re-implementations of the prompt path leave
  the token stream where MiaAI-Lab's 1.1.4-era patches left it. Upstream asserts "the same tokens" for each
  (kernel contract, exact comparisons) and MiaAI-Lab asserts the same for theirs; both cannot be checked from
  the trees, and the check that settles it is a logprob/decisions comparison on the served model — the same
  instrument this rig already uses (`/v1/decisions`, jsd and top-1 agreement against a reference arm). That
  run is owed, and it is the first thing to do once the endpoint can be taken.

## 5. Memory with the new defaults

`mem_need_gib 262144` at bf16 with vision is **110 GiB** (`scripts/config.sh`: weights 81 + one window 7.5 +
8 scratch + 2 vision + 9 reserve + 2), so a start needs a Spark with nothing else large resident — the same
constraint as before, since the weights decide it. `kv_gib 262144` reads 8 GiB at bf16 against 5 GiB at fp8.
`PARALLEL=2` is measured, not arithmetic: at 262,144 tokens with the bf16 cache the engine refused
`--parallel 3` before listening — 23.43 GiB for its streams against a 21.55 GiB CUDA budget, 7.81 GiB a
window — and served two (`max_batch_size 2`, 78.7 GB active). An earlier draft of this page argued three from
MiaAI-Lab's 24.3 GiB pool figure; the engine's own number is what counts, and its admission check refuses a
`PARALLEL` whose windows do not fit, so a wrong default fails at startup rather than at the first request.

## 6. Measured on the pin, 2026-10-28

Built with `zig build -Dnvcc=... -Doptimize=fast fatbins native` (exit 0) and served on one GB10 Spark:
NVIDIA's FP4 QAD checkpoint, `--context 262144 --parallel 2 --kv-cache bf16 --thinking
--reasoning-effort xhigh --max-tokens 32768 --temperature 0`, no vision, one client. The engine reported
`version 1.1.8`, `max_batch_size 2`, 78.7 GB active.

**The defaults in `scripts/config.sh` were wrong and are now measured.** The first start was refused before
the server listened: "`--parallel 3` needs 23.43 GiB for its streams at a 262144-token window, and the CUDA
memory budget leaves 21.55 GiB" — 7.81 GiB a window, so two fit. `PARALLEL` is now 2, and §5 says why.

Prefill, fresh prose (the tool counts only the prompt's own tokens, not the 72-token template it caches):

| prompt | prefill s | tok/s |
| --- | --- | --- |
| 6,311 | 3.62 | 1,745 |
| 12,596 | 7.04 | 1,788 |
| 25,163 | 13.78 | 1,827 |
| 50,268 | 30.45 | 1,651 |
| 100,532 | 56.54 | 1,778 |

Decode, 256 tokens, median of 5: greedy code **42.1 tok/s**, sampled chat **48.0 tok/s**.

### The MXFP8 lane, with and without

Same binary (`6e5f454`), same flags, only `TF_FLASHNEXT_MXFP8` differs. The widened arm logs one
`MXFP8 off, widening {n}x{k} e4m3 to bf16` line per plane (240 of them) and holds 3.0 GB more device memory,
which is the check that the switch is the only variable.

| | stored codes (default) | widened bf16 (`=0`) |
| --- | --- | --- |
| active memory | 78.72 GB | 81.57 GB |
| prefill tok/s, 6.3k / 12.6k / 25k / 50k / 100k | 1,676 / 1,679 / 1,696 / 1,792 / 1,711 | 1,646 / 1,754 / 1,737 / 1,666 / 1,695 |
| decode greedy | 42.6 | 41.5 |
| decode sampled | 47.0 | 42.3 |

So the lane is worth **about 10% of sampled decode and 3.0 GB**, and nothing measurable on prefill: those
planes are re-read every token and a prompt is a different balance. An earlier pair of runs came out identical
because the served binary was the pin's, which has no switch at all — the branch has to be built for the arm to
exist, and the widening lines in its log are what proves it did.

### Against the arm it replaces

The endpoint this work replaces (v1.1.6 fork build, MIA engine, `TF_FLASHNEXT_DEPTH=15
TF_FLASHNEXT_CONFIDENCE=-0.4 TF_FLASHNEXT_PRODUCT_STREAMS=2`, three streams, vision helper, 81.4 GB active)
was measured on the same rig harness, 512-token replies, one client: prose 10.68 s, code 9.95 s. The pin on the
same harness: prose 14.52 s, code 9.41 s — **26% slower on prose, 5% faster on code**. The difference is the
draft chain's rule, not the engine: 1.1.8 reads `TF_FLASHNEXT_DRAFTS` (default 6) and
`TF_FLASHNEXT_CONFIDENCE` (default 0.60), where the old arm drafted deeper and never stopped early. Setting them
on the pin (`TF_FLASHNEXT_DRAFTS=15 TF_FLASHNEXT_CONFIDENCE=-0.4`) is the next run, and it needs the GPU: it is
the cheapest way to know whether the pin loses anything at all.

## 7. What is still on the table

### The MXFP8 lane, and its A/B arm

`kinetick-labs/TensorFold` `feat/mxfp8-switch-1.1.8` (`6e5f454`) adds `TF_FLASHNEXT_MXFP8=0`, which widens the
checkpoint's e4m3 planes to bf16 on the host instead of serving them as stored codes. **Upstream 1.1.8 already
serves them natively** — `cuda_projection.zig:mxfp8` keeps the codes and the group-32 UE8M0 scales, reached from
`cuda_projection_load.zig` for the attention and DeltaNet stitches a FP4-Local checkpoint carries (and for the
shared expert and the prompt key/value projections) — and it has no switch to turn that off. So the branch is
not a port of anything: it is the *other* arm, because a bench with and without the lane needs both.

Both arms give the same numbers: an e4m3 code times a power of two keeps its three mantissa bits inside bf16's
seven, so widening is exact (the only values that cannot round-trip are f32 subnormals below 2^-126, which no
checkpoint's scales, near 127, produce). What differs is bytes: the widened planes are twice the size, which
the start's memory estimate and the sequence pool both feel.

    TF_REF=6e5f45424 IMAGE=tensorfold-qwen38:mxfp8-off scripts/prepare.sh
    IMAGE=tensorfold-qwen38:mxfp8-off ./start.sh                      # with the lane (upstream's default)
    IMAGE=tensorfold-qwen38:mxfp8-off TF_FLASHNEXT_MXFP8=0 ./start.sh # without it, widened

`start.sh` forwards every `TF_FLASHNEXT_*` variable into the container already, so no script changes: two
images is the only way, since the switch lives in the engine.

### MiaAI-Lab's remaining patches

Wanting their remaining wins means porting, in this order of value per line of risk:

1. `0017` the stop rule — draft rows only, a few percent of greedy code decode, no reply change.
2. `0012`, `0016` — keep-late and the shared side stream/lane-batched DeltaNet; decode rounds, claimed
   bit-identical, and the largest remaining multi-stream work.
3. `0007`, `0008`, `0010`, `0011` — sampled requests in the batched draft pass and the candidate/host-draw
   machinery; matters for chat traffic (temperature > 0), not for greedy.
4. `0003` + `0005` — fp8 KV. Only worth porting if five full windows are wanted; it is the lossy one, and
   porting it onto the paged caches is the largest job of the four.

## 8. Limits

- What §6 reports was measured here: one build, one serve per arm, one client, single runs, +-4% noise on
  this rig. **No quality run was made** — `tools/exact.py`, `toolcheck.py`, `visioncheck.py`,
  `prompt_reuse.py` and the 200k needle are all owed on this pin, and vision itself was off in every run
  above (the two patches this fork keeps are vision patches, so they are unexercised by these numbers).
- Every other throughput figure in this document is quoted from MiaAI-Lab's README/CHANGELOG or upstream's,
  at their defaults, and is not comparable with §6's (different harness, checkpoint and stream count).
- The absorbed/port verdicts are read off the tag's tree (file presence, knob names, changelog claims). A
  file of the same name is not proof of identical arithmetic; that is what the decisions comparison in §4
  is for.
