# Multi-session bench — perf/zig-exl3-decode @ 6a9ae92f (older run; current numbers in README)

One GB10, `scripts/serve.sh` settings (bf16 KV, `TF_FLASHNEXT_PREFILL_ROWS=4096`, cap 4 / −0.6, 95k draft vocab),
one server per `--parallel` level, measured 2026-10-12.

Tool: `tools/bench_concurrent.py` from the TensorFold checkout,
`--levels N --tokens 256 --reps 3 --temperatures 0 --alone` (N identical requests start together), then
`--mixed --alone` (stream i gets prompt i % 2, fib/chat interleaved; mixed cells are sampled at T=1 with a
fixed seed per stream). `--alone` runs every request alone first and compares each concurrent reply's
`token_sha` with it. Medians of 3 reps. Aggregate = all streams' tokens after their first / wall span;
steady = while every stream is live.

| `--parallel` | Context | Prompt | Aggregate tok/s | Steady tok/s | Per-stream tok/s | TTFT max | Concurrent = solo |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 262,144 | fib | 63.4 | — | 63.4 | 0.09 s | 3/3 |
| 1 | 262,144 | chat | 49.6 | — | 49.6 | 0.13 s | 3/3 |
| 2 | 262,144 | fib ×2 | 110.0 | 112.3 | 55.5 | 0.18 s | 6/6 |
| 2 | 262,144 | chat ×2 | 88.6 | 90.5 | 44.4 | 0.27 s | 6/6 |
| 2 | 262,144 | mixed (T=1) | 67.2 | 76.6 | 39.4 | 0.22 s | 6/6 |
| 3 | 131,072 | fib ×3 | 151.8 | 157.4 | 51.5 | 0.26 s | 9/9 |
| 3 | 131,072 | chat ×3 | 119.9 | 124.3 | 40.8 | 0.39 s | 9/9 |
| 3 | 131,072 | mixed (T=1) | 82.7 | 96.1 | 34.5 | 0.31 s | 9/9 |
| 4 | 98,304 | fib ×4 | 181.4 | 190.4 | 46.5 | 0.35 s | 12/12 |
| 4 | 98,304 | chat ×4 | 123.2 | 129.3 | 31.5 | 0.53 s | 12/12 |
| 4 | 98,304 | mixed (T=1) | 94.7 | 111.1 | 29.6 | 0.44 s | 12/12 |

No request failed, no OOM, no error in any server log. Every concurrent reply had the same `token_sha`
as the same request served alone.

**Memory limits (bf16 KV, 12 GiB kept free).** The server refuses at startup rather than OOM:

- `--parallel 3` at 262,144: needs 20.74 GiB for streams, 16.51 GiB available → refused.
- `--parallel 4` at 262,144: needs 27.65 GiB, 17.32 available → refused.
- `--parallel 4` at 131,072: needs 14.24 GiB, 14.20 available → refused (by 0.04 GiB).

So: `--parallel 2` at the full 262k window, `--parallel 3` at 128k, `--parallel 4` at 96k
(or `--kv-dtype fp8`, which halves KV but is not covered by these shas).
