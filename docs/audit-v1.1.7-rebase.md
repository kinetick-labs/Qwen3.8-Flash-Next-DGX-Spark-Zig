# The v1.1.7 rebase: what MiaAI-Lab's series still adds, and what its speed costs in quality

Supersedes the v1.1.4 pass in [`audit-v1.1.4-rebase.md`](audit-v1.1.4-rebase.md), which stays as the
record of what this repository's own recipe (the EXL3 4.05 bpw arm) had added and what a TensorFold
v1.1.4 base retires.

The base is now **TensorFold v1.1.7** (`c5953415e`), not v1.1.4 plus someone else's seventeen patches.
That is a correction, not a preference: the prompt path MiaAI-Lab's series was written for is *in the
engine* from 1.1.5 and 1.1.7, so replaying it on v1.1.7 would re-implement what upstream already ships
with its own exactness gates.

## 1. What this fork pins

`kinetick-labs/TensorFold` `integration/1.1.7-vision` (`1000c39c169a3f973464966d9999409b0e3b4b96`):
v1.1.7 plus **two** patches, both from MiaAI-Lab's series and both lossless:

- `0001` — the NVIDIA FP4 checkpoint's quantized vision tower (MXFP8 attention, NVFP4 MLP and merger)
  decoded to bf16 at load, so `--vision` works on `nvidia/Qwen3.8-Flash-Next-FP4-Local`;
- `0002` — the per-request image-token budget at 16384 instead of 4096.

Both apply cleanly on every tag from v1.1.4 to v1.1.7, and neither has an upstream equivalent:
`zig/src/vision/qwen_cuda_dequant.zig` (0001's new file) does not exist at v1.1.7, and v1.1.7 still caps
a request's image rows at 4096. Provenance is by content: the branch's tree hash equals a pristine
v1.1.7 worktree after `git apply` of the two — `38165d03c8` on both sides.

## 2. The other fifteen patches: absorbed, or not carried

Measured on a pristine v1.1.7 worktree: each patch tried on its own, then the series in filename order.
Only `0001` and `0002` land; `0003` starts the conflicts because v1.1.5 rewrote the attention cache
(pages, core pool) the patch touches.

| patch | verdict on v1.1.7 | evidence |
| --- | --- | --- |
| 0001 vision FP4 tower | **carried** | new file `zig/src/vision/qwen_cuda_dequant.zig`, absent upstream |
| 0002 image tokens 16384 | **carried** | upstream still caps at 4096 in `qwen_cuda_media.zig` |
| 0003 `--kv-cache fp8` for Flash Next | not carried — **the lossy one** (see §3) and needs a port | upstream has no `kv8` in `zig/src/families/flashnext` at any tag up to v1.1.7; the patch no longer applies since 1.1.5's paged caches |
| 0004 prompt expert tile | absorbed upstream | `zig/kernels/cuda/nvfp4_experts_tile.cu` + `zig/src/cuda/experts_tile.zig` at v1.1.7, kernel-contract entries, changelog "NVFP4 experts on prompt tiles" (1.1.5) |
| 0005 fp8 reader + 4-bit MTP cache | not carried (tied to 0003) | drafts-only either way; no value without the fp8 cache |
| 0006 prefill speed | partly absorbed | 1.1.5: no zero fill of ModelOpt's read-back, hyper-connection normalize over 256-column slices, half tiles; remainder would need a port |
| 0007 sampled requests in the batched pass | not carried | no `sampled` path in v1.1.7's `cuda_lanes_draft.zig`; port candidate |
| 0008 GPU top-k candidates | not carried | no `fn_row_cand`/`row_cand` at v1.1.7; port candidate (its candidate set is the request's `top_k`, i.e. the same set the host draw keeps) |
| 0009 round-cost model | not carried — off by default | zero value at the default, and `cuda_round_costs.zig` has no upstream home |
| 0010 n-gram gather of all lanes in one pass | not carried | port candidate |
| 0011 host draws on worker threads | not carried | port candidate |
| 0012 DeltaNet kept-rows log | not carried | port candidate |
| 0013 n-gram pread reader + chunk tail | absorbed upstream | `zig/src/families/flashnext/cuda_ple_reader.zig` and `cuda_ple_ahead.zig` at v1.1.7, driven by `TF_FLASHNEXT_PLE_IO` and `TF_FLASHNEXT_WARM_AHEAD` (1.1.7) |
| 0014 prompt hyper-connection fusion | partly absorbed | 1.1.5: hyper-connection normalize and dense mix work; the fused-kernel set is upstream's now |
| 0015 MX128 two-stage prompt tile | overlapping, not equal | 1.1.7 has a persistent 128x256 tile (#1350) but the changelog calls Flash Next NVFP4 *unchanged* by it, so this one may still add something — a port and a bench, not an assumption |
| 0016 shared expert side stream + lane-batched DeltaNet | not carried | no `cuda_batch_side.zig` upstream; port candidate |
| 0017 draft running-product stop rule | not carried | v1.1.7 keeps the per-draft confidence floor; the rule changes draft rows only |

## 3. The speed/quality trade

Short version: **the only quality the recipe trades is the KV cache, and the trade is in the recipe's
defaults, not in the engine.**

- **`KV_DTYPE=fp8` is lossy by construction.** An fp8 cache stores keys and values as e4m3 codes with a
  power-of-two scale per position and KV head, and attention reads them widened back to bf16: the bf16
  kernels' arithmetic on rows rounded to e4m3. MiaAI-Lab's own README prices it — "about 1.84x the pool of
  bf16, and lossy (~98.8% top-1 agreement with a bf16 cache, so a free-running reply can differ)" — and
  their published five-stream tables at 262,144 tokens are taken with it, because 5 x 4.4 GiB fits where
  5 x 7.5 GiB does not. That is the whole of their headline speed story as far as quality goes: their
  README's default *is* the lossy configuration.
- **So this fork defaults to bf16 and pays in streams, not in bits.** `KV_DTYPE=bf16`, `PARALLEL=3` at
  262,144 tokens (7.5 GiB a window against the ~24.3 GiB pool the engine reports after load); `PARALLEL=5`
  is available and honest with `KV_DTYPE=fp8`, and `scripts/config.sh` says so in one line rather than
  burying it. Upstream's other CUDA families take `--kv-cache fp8` the same way: available, off by default.
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

## 4. Memory with the new defaults

`mem_need_gib 262144` at bf16 with vision is **110 GiB** (`scripts/config.sh`: weights 81 + one window 7.5 +
8 scratch + 2 vision + 9 reserve + 2), so a start needs a Spark with nothing else large resident — the same
constraint as before, since the weights decide it. `kv_gib 262144` reads 8 GiB at bf16 against 5 GiB at fp8.
`PARALLEL=3` is arithmetic on the engine's own reported pool, not a measurement: the admission check refuses
a `PARALLEL` whose windows do not fit, so a wrong default fails at startup rather than at the first request.

## 5. What is still on the table

Wanting their remaining wins means porting, in this order of value per line of risk:

1. `0017` the stop rule — draft rows only, a few percent of greedy code decode, no reply change.
2. `0012`, `0016` — keep-late and the shared side stream/lane-batched DeltaNet; decode rounds, claimed
   bit-identical, and the largest remaining multi-stream work.
3. `0007`, `0008`, `0010`, `0011` — sampled requests in the batched draft pass and the candidate/host-draw
   machinery; matters for chat traffic (temperature > 0), not for greedy.
4. `0003` + `0005` — fp8 KV. Only worth porting if five full windows are wanted; it is the lossy one, and
   porting it onto 1.1.5+'s paged caches is the largest job of the four.

## 6. Limits

- No build, serve, bench or quality run was made for this document beyond the two `git apply` matrices and
  the tree-hash identity in §1. `PARALLEL=3`, the 24.3 GiB pool and every throughput figure above are
  quoted from MiaAI-Lab's README/CHANGELOG or computed from `scripts/config.sh`, not measured here.
- The absorbed/port verdicts are read off v1.1.7's tree (file presence, knob names, changelog claims). A file
  of the same name is not proof of identical arithmetic; that is what the decisions comparison in §3 is for.
