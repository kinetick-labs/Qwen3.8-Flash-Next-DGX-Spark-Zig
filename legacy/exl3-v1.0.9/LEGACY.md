# The EXL3 4.05 bpw arm, on the engine that served it

This directory is the recipe this repository carried **before** the v1.1.4 rebase, kept verbatim so the
turboderp EXL3 4.05 bpw arm stays reproducible. Nothing here is served by the top-level `start.sh`: it
builds a different engine revision and serves a different checkpoint.

- **What it serves.** `turboderp/Qwen3.8-Flash-Next-exl3`, branch `4.05bpw_h6_ng6` (rev `55a732e0`), through
  `tensorfold-native` built from our fork at `0d1c730` — TensorFold v1.0.9 plus our EXL3 enablement
  (kinetick-labs/TensorFold PRs #13, #14, #18).
- **How it is built and run.** `scripts/build.sh` (source build: `zig build native -Dnvcc=… -Dsm=121`),
  then `scripts/serve.sh` in the foreground; `scripts/add-kv8.sh` splices the FP8-KV Triton cubins of
  `kernels/kv8/` into an sm121 kernel set first, and `scripts/vision-helper.sh` provides the Python vision
  helper when `VISION=1`.
- **Its numbers.** `README.md` (the recipe's) and `bench/results.md` (the multi-session run with its memory
  limits): decode 62.1 / 49.7 tok/s solo, 178.2 / 139.9 aggregate at four streams, prefill 1,079 tok/s at a
  14k prompt — all with `tools/bench_concurrent.py` at temperature 0, which is **not** the instrument behind
  the v1.1.4 tables in the top-level README (sparkDash, NVIDIA's FP4 checkpoint). The two are not
  comparable; see `docs/audit-v1.1.4-rebase.md`.
- **Why it is not rebased.** TensorFold v1.1.4 serves no EXL3 checkpoint (`docs/recipes/exl3.md`: "No model
  serves an EXL3 checkpoint in this release"), and its Flash Next CUDA engine was rewritten under our
  work twice over. Porting native EXL3 onto it is a project of its own, not a re-pin.

The files are exactly as they were on `main` before the rebase (`git show <pre-rebase-main>:scripts/…`).
