# The v1.1.4 pass: what MiaAI-Lab's new recipe brings, and which of our fixes it retires

> **Superseded** by [`audit-upstream-rebase.md`](audit-upstream-rebase.md). This page is the first pass, taken
> when the base was going to be TensorFold v1.1.4 plus MiaAI-Lab's seventeen patches; the base moved to
> upstream v1.1.8 because 1.1.5 and 1.1.7 ship the prompt path and the n-gram reader that series was written
> for. Everything below about **our own** additions (the EXL3 arm's kv8 cubins, vision-helper container,
> media fix, profiles, prefill rows, product stop, build/bench scripts) still stands and is not restated there.

Audit for rebasing this recipe onto
[MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)
at `fd3f43e` (branch `main`, 2026-10-27): TensorFold v1.1.4 `eab85828` plus their seventeen patches
`patches/0001` to `0017`. Our previous recipe pinned our own fork of the Zig engine at `0d1c730`
(TensorFold v1.0.9 + our pull requests #13, #14, #18) and served turboderp's **EXL3 4.05 bpw** pack.

Two things are settled by reading the trees, and one by applying the patch series; nothing here was
re-measured on a GPU.

## 1. The two recipes do not serve the same checkpoint, so their numbers are not ours to beat

|  | our previous recipe | theirs (v1.1.4 + 0017) |
| --- | --- | --- |
| Checkpoint | `turboderp/Qwen3.8-Flash-Next-exl3` `4.05bpw_h6_ng6`, ~107 GB | `nvidia/Qwen3.8-Flash-Next-FP4-Local`, ~96 GiB (NVFP4 QAD) |
| Engine | v1.0.9 + our EXL3 work (our fork) | v1.1.4 + 17 patches (upstream) |
| Instrument | `tools/bench_concurrent.py` from the checkout | sparkDash Decode/Prefill |
| Published decode | 62.1 / 49.7 tok/s solo (fib / chat); 178.2 / 139.9 aggregate at 4 | 55.6 prose / 110.5 code solo; 145.1 / 192.9 aggregate at 4 |
| Published prefill | 1,079 tok/s (14k prompt) | 2,839 tok/s (16k prompt), 3,050 at 4k |

So the claim "better results" is, on the numbers both READMEs carry: **prefill, by a wide margin**
(2.8-3.0k against 1.08k), and **decode, broadly level** (they beat us on their strongest cell, we beat
them on ours). A different checkpoint and a different benchmark client are in each column, so the
comparison is indicative only, in both directions.

Their own CHANGELOG is the sharper reading: against the previous `db281878` engine build — which is the
lineage our fork already carries (`feat/fp4-local*`, the `TENSORFOLD_FLASHNEXT_ENGINE=mia` engine, the
one this rig serves) — the v1.1.4 build is *"10 to 15% behind the previous build at 5 streams and 10 to
15% behind on prefill, and ahead at 1 stream"*. So on the same checkpoint, their new recipe is a
single-stream win and a multi-stream and prefill loss against what we already run. What is genuinely new
to us is their **prompt path** (patches 0004, 0006, 0014, 0015) and the db281878-era prefill work it
re-expresses, not a faster multi-stream decoder.

## 2. Their patch series is a series, and it applies only to a pristine v1.1.4

- On a pristine `v1.1.4` worktree, `git apply` in filename order: **17/17 clean**.
- `git apply --check` each patch on its own (against the same pristine tree or against ours) reports
  "conflict" for ten of them: 0005, 0008, 0010, 0011-0017 edit regions 0003/0004/0006 created. A series
  has to be replayed cumulatively; a per-patch matrix says nothing.
- On our own v1.1.4 rebase branch (`feat/fp4-local-v1.1.4`), the cumulative replay collides from `0003`
  on (13 unmerged files under `--3way`): our rebase edits the same stock files the series edits
  (`zig/kernels/cuda/flashnext_attn.cu` +134/-…, `cuda_batch.zig`, `cuda_mtp.zig`, …).

Hence the shape this fork takes: an integration branch on our engine fork carries the seventeen patches
as seventeen commits, and the recipe pins that revision instead of replaying `patches/*.patch` at image
build time.

`integration/1.1.4-mia17` (kinetick-labs/TensorFold) was built exactly that way, one commit per patch,
messages crediting MiaAI-Lab, and checked by content: its tree hash equals the tree of a pristine `v1.1.4`
worktree after all seventeen patches were applied (`git write-tree` on both sides: identical,
`48745f0d9bc367211b9993af2874595ce4c25eb8`). The recipe then points `TF_REPO`/`TF_REF` at that branch
and `patches/` is empty — a zero-delta overlay would otherwise re-apply what the pin already carries.

## 3. Which of our fixes are still needed

Read as: what did *our* recipe add on top of the engine, and does v1.1.4 + 0017 already carry it?

| Our fix | Where ours lived | Verdict on v1.1.4 + 0017 | Evidence |
| --- | --- | --- | --- |
| FP8 KV from a captured Triton set: `kernels/kv8/` (5 cubins), `scripts/add-kv8.sh`, `serve.sh` refusal | `kernels/kv8/`, `scripts/add-kv8.sh` | **Retired, not needed.** v1.1.4's KV cache is the engine's own CUDA glue and their 0003 adds fp8 to it; the image embeds its kernel set, so there is no `aot.json` to splice cubins into and no `_attn_prep8` to capture | 0003's own doc hunk: "`--kv-cache fp8` (one rank, the engine's own glue; not with `--slide`)"; `scripts/prepare.sh` builds `fatbins native` and labels the image `tf.kernels=embedded`; `zig/kernels/cuda/flashnext_attn.cu` carries the fp8 reader after 0003, and no file in v1.1.4 names `attn_prep8` |
| Vision through a Python helper **container** (`scripts/vision-helper.sh`, `VISION_IMAGE`, extra 4 GiB reserve, `TENSORFOLD_MEMORY_RESERVE_GIB=11.5`) | `scripts/vision-helper.sh`, `serve.sh` | **Retired, not needed.** v1.1.4 offers `--vision` from the CUDA host itself and their recipe runs the helper inside the PyTorch image it already ships; 0001/0002 give the FP4 checkpoint its tower and a 16384-token image budget | `git show v1.1.4:zig/src/native/cuda.zig` line 455: `.vision = if (@hasField(@TypeOf(loaded), "vision")) vision_offer.offer(loaded.vision, o.vision) else null`; their `scripts/config.sh`: `VISION=1` passes `--vision` and "rank 0 starts the Python helper" |
| The engine fix advertising media for Flash Next under `--vision` (our PR #18, `zig/src/native/cuda.zig`) | our fork, `fix/zig-flashnext-media` | **Retired, not needed.** The refusal it fixed is gone: v1.1.4 builds the host's `Info` with `vision_offer`, and our patch applied to *our* host file (`cuda_mia.zig`), which is not in the v1.1.4 path at all | the `vision_offer` line above; v1.1.4 also carries `cuda_flashnext_refusal_test.zig` ("Flash Next takes --vision at family admission") |
| `PROFILE=single\|multi`: bf16 KV, draft depth 4 / fp8 KV, depth 3, 4 streams at 262k | `scripts/serve.sh` | **Replaced.** The profile pair was an EXL3 acceptance-rate choice; on v1.1.4 the depth is a stop rule (0017), the engine defaults are 6 drafts at confidence 0.60, and 5 streams at 262k fit with fp8 KV | their `config.sh`: `PARALLEL=5` with the pool math, `TF_FLASHNEXT_DRAFTS=6`, `CONFIDENCE=-0.4`, `PRODUCT_STREAMS=5`; 0003 doc hunk: "`TF_FLASHNEXT_DRAFTS` (1 to 15, default 6) and `TF_FLASHNEXT_CONFIDENCE` (default 0.60)" |
| `TF_FLASHNEXT_PRODUCT_STREAMS=2` and the running-product stop (`confidence -0.6`) | `scripts/serve.sh` | **Carried by them: patch 0017.** Same rule, their numbers: the db281878 rule with a −0.4 floor, up to 5 drafting streams, +4.3% greedy code at 5 streams, +10.2% at 3, +13.6% at 1 | `patches/0017-draft-chain-running-product-stop-rule.patch`; their CHANGELOG entry for 0017 |
| `TF_FLASHNEXT_PREFILL_ROWS=4096` | `scripts/serve.sh` | **Replaced.** 3072-row prompt chunks with a tail rule, plus the prompt tiles of 0014/0015 | 0006 and 0013 (chunk tail under 512 rows), 0015 (`PROMPT_GEMM2`) |
| `scripts/build.sh`: source build, `zig build native -Dsm=121 -Dnvcc`, `TF_REF=0d1c730` | `scripts/build.sh` | **Replaced by their image build**; our pin becomes the integration branch above | their `scripts/prepare.sh` |
| `scripts/bench.sh` + `bench/results.md` (`bench_concurrent.py`) | `scripts/bench.sh` | **Kept as an instrument, not as a comparison.** It is what our figures were measured with; the published v1.1.4 figures are sparkDash's, so a number from one is not a baseline for the other | `bench/results.md` header; their README "Performance" preamble |
| Our EXL3 engine work: native CUDA EXL3 (PR #5), 4-bit draft head (PR #14), faster EXL3 prefill grouping (PR #13), EXL3 packs in the native backend (PR #15) | our fork, `feat/zig-exl3*` | **Still ours, and not portable to this base.** v1.1.4 serves **no** EXL3 checkpoint, and its Flash Next CUDA engine was rewritten twice since the tree our work was written against | `docs/recipes/exl3.md` at v1.1.4 and at `main`: "No model serves an EXL3 checkpoint in this release", with the CUDA operators "in review" |
| Draft vocabulary of 95,111 ids (PR #14's subject) | our fork | **Not a delta.** Upstream and our fork ship the same file | `draft_vocab_default.txt` is 79,591 lines on both; the 95,111 figure appears only in a comment |

Net: of the nine things our recipe added, **seven are retired or replaced by the v1.1.4 base**, one is kept
(the bench instrument, which is not a baseline for the published figures), and one — the EXL3 engine stack —
has no home there at all.

## 4. What happens to the EXL3 arm

It is not deleted, it is pinned where it still runs: `legacy/exl3-v1.0.9/` keeps the previous recipe
verbatim — its `scripts/`, its `kernels/kv8/`, its measurements, its pin `0d1c730` — so the EXL3 4.05 bpw
arm stays reproducible on the v1.0.9 engine it was measured on.

Porting it to v1.1.4 would mean re-implementing native EXL3 on an engine that has changed under it
(two releases, a rewritten CUDA path, and a prompt pipeline the seventeen patches rebuild), for a
checkpoint upstream does not qualify; that is its own project, not part of this rebase.

## 5. Limits of this audit

- Nothing was re-measured: no build, no serve, and no bench ran for this document. The only dynamic
  checks are the `git apply` matrix and the tree-hash identity above.
- Their sparkDash rows are theirs, on their Spark, at their sampling settings; our decode rows were taken
  with `bench_concurrent.py` at temperature 0 on the EXL3 pack.
- The seventeen patches were read as a series and applied as one; their individual interactions with our
  own engine branch (what would conflict, and how) were measured on `feat/fp4-local-v1.1.4` but not
  resolved there — this fork does not carry that branch.
