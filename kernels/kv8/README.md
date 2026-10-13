# FP8 KV cache Triton kernels (sm_121)

`_attn_prep8` (3 variants) and `_chunks8` (2 variants), TP=1: the `--kv-dtype fp8` attention kernels that the
TensorFold sm121 Triton sets used by this recipe do not contain. `scripts/add-kv8.sh` merges them into a copy of
your kernel set.

- **Source:** extracted unchanged from `ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold:zig-db28187-4144696f21e3`,
  `/opt/tensorfold/share/tensorfold/cuda/sm121/` (generator `tools/zig/flashnext_aot.py`, target cuda/121).
  Python source `tensorfold.families.qwen4_exp.cuda.kv8`; spec rule `tools/zig/flashnext_kv8_spec.py`.
- That image's bf16 `_attn_prep` / `_chunks` cubins have the same hashes as the recipe's set, so these
  were compiled from the same sources and Triton.
- `kv8.json` keeps each entry's `cubin_sha256`; the script checks it before merging.
- Rebuilding them yourself: `flashnext_kv8_spec.py` over `zig/tests/cuda/flashnext/kernels.json`, then
  `flashnext_aot.py build --only kv8` in a PyTorch/Triton image with the TensorFold Python tree.

TensorFold is Apache-2.0.
