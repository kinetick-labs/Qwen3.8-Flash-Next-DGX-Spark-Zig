# Bench — kinetick-labs/TensorFold main 80071f40c (Spark GB10, this recipe)

3 distinct agents (A zig review ~43k, B French docs ~25k, C CUDA ~65k first prompts) x 4 turns, 256 tokens a reply,
temperature 0, fp8 KV, `--parallel 3`, 262,144 context, 12 GiB prompt cache. Tool: the multi-agent script used for
all of the fork's prefill work (`ma.py`); TC2 = 7 strict-tool / json_schema checks.

| config | turn-1 avg | A/B/C turn-1 | later turns avg/max | decode avg/min tok/s | cache cap | solo prefill 14k | hits |
|---|---|---|---|---|---|---|---|
| fp8 base, no slicing | 38.6 s | 21.4/32.8/61.7 | 0.74/1.22 s | 22.5/4.7 | 11753 MiB | 2328 tok/s | 9/12 |
| sliced, 3 prompts interleaved (busy 4) | 65.7 s | 70.1/56.6/70.2 | 0.83/1.16 s | 27.8/20.1 | 11553 MiB | 2313 | 9/12 |
| **this recipe** (srf, busy 1, rows-busy 1024, hold 3 s), 992d26f77 | **64.0 s** | 22.4/58.5/111.0 | **1.06/1.34 s** | 44.1/**31.4** | 11521 MiB | **2313** | 9/12 |
| this recipe on main (rebased on upstream 45fcab6c2), 6bcf5588e = 80071f40c minus test-only commits | 63.6 s | 22.3/58.2/110.4 | 1.06/1.35 s | 44.5/31.6 | 11399 MiB | 2299 | 9/12 |

DFlash2 single-stream code decode ~100-114 tok/s (MTP alone ~73).
Exactness: on outputs only (greedy): every multi-agent reply hash and the single-stream hashes match the fp8 base;
logits were not compared bit for bit (smaller busy chunks move the chunk cut, which may change bits).


# Bench — kinetick-labs/TensorFold main 17334bddd (upstream d6c4cce60 / 1.2.8, Spark GB10, this recipe)

Single-stream, temperature 0, same prompts as the deploy gate against `58ece2237`. Decode hashes are identical.
Prefill 14k is the mean of the last two of three reps (first rep is a warmup).

| pin | prefill 14k | decode fib | decode chat | decode fr | hashes (fib / chat / fr) |
|---|---|---|---|---|---|
| 58ece2237 (previous main) | 2,347 tok/s | 112.6 tok/s | 47.4 tok/s | 51.1 tok/s | c93c652df0a0 / 49127283b123 / 4de58665d0fb |
| **17334bddd (this pin), candidate gate** | **2,274** tok/s (−3.1%) | **113.1** | **47.3** | **50.6** | same |
| 17334bddd, prod :8083 after cutover | ~2,350 tok/s | 112.2 | 47.3 | 50.7 | same |

Vision, `tool_choice: required` and `json_schema` (TC2 7/7) passed on this pin. A client leaving mid-prefill no longer
crashes the engine (same drain fix as 58ece2237). `TENSORFOLD_PREFILL_POLICY=srf` is accepted again.

A concurrent 3-agent session on the candidate kept the same reply hashes as 58ece2237 (order differed because the
requests ran together) and was not used as a speed gate. Historical multi-agent timings for earlier pins remain in the
table above.


# Bench — kinetick-labs/TensorFold main 082aee98a (compute lane + runner, Spark GB10, this recipe)

Deploy gate against the `17334bddd` baseline, single stream, temperature 0 (prefill: mean of the last two of three reps).

| pin | prefill 14k | decode fib | decode chat | decode fr | hashes (fib / chat / fr) |
|---|---|---|---|---|---|
| 17334bddd (baseline) | 2,276 tok/s | 112.2 | 47.3 | 50.7 | c93c652df0a0 / 49127283b123 / 4de58665d0fb |
| **082aee98a, prod :8083** | **2,291** tok/s | **112.7** | **47.3** | **50.9** | same |

Compute lane suite (15 exact math/logic + 4 coding tasks run against asserts; thinking on, no client tools, 2 at a time):

| | pass | tokens a task (reasoning) | wall a task | cut by length |
|---|---|---|---|---|
| 17334bddd, no lane | 14/19 | 1,984 (1,929) | 36.7 s | 5 |
| lane + runner, DFlash2 off | 19/19 | 314 (152) | 7.3 s | 0 |
| lane + runner + DFlash2 (candidate) | 19/19 | 314 (152) | 6.8 s | 0 |
| 082aee98a, prod | 19/19 | 314 (152) | 6.3 s | 0 |

Agents with their own tools (Hermes-like, 10 requests / 6 tool loops): the built-in tool is not offered to them, so
their prompts match the lane-off server's token for token; hermes2 completion 352 tokens and 6.1 s a task (lane-off
350 / 5.8 s; offering python next to their tools had cost 432 / 7.1 s and 2 failures). Multi-agent session (`ma.py`,
3 agents x 2 turns): later-turn TTFT avg 16.0 s, decode avg 33.5 / min 26.7 tok/s.

Vision on 082aee98a: text, counting, chart, table, photo and multi-image checks 14/15 (the 15th is an agent with a
terminal tool that computed the table total through the tool, as asked); a 5184x2920 MPO camera photo is accepted
(4,117 prompt tokens, 4.4 s) and read correctly; animated PNG and WebP are refused.


# Bench — prod 082aee98a, 2026-11-16 (Spark GB10, this recipe)

Single-stream restore checks of the prod server after each of today's 8 test windows (temperature 0, same prompts as
the deploy gate). `/health` answered 200 96–99 s after each start.

| check | decode fib | decode chat | decode fr | hashes (fib / chat / fr) | TC2 |
|---|---|---|---|---|---|
| range over 8 restores | 112.6–113.6 | 47–48 | 49.8–51.8 | c93c652df0a0 / 49127283b123 / 4de58665d0fb every time | 7/7 every time |

Agent jobs on prod (Mercury): `veille-colombages` weekly run 2026-11-13: 65 min, 3.08M tokens in / 79k out,
67 steps, 78 tool calls; `veille-dgx` 2026-11-14: 125 s, 52.7k / 1.8k, 7 steps; 2026-11-15: 127 s, 37k / 1.9k,
5 steps, 4 tool calls.

## Concurrency c1–c4 (decode, 2026-10-17, earlier FP4-Local build, not this pin)

4 lanes, drafts on, before the DFlash2 drafter and srf; 256 tokens a stream, temperature 0, every output equal to the
same request alone. Prefill under concurrency was not measured; on this pin only c1 (above) and the 3-agent session
(`ma.py`: decode avg 33.5 / min 26.7 tok/s a stream) exist.

| streams | code total / per stream tok/s | chat total / per stream tok/s | TTFT max code / chat |
|---|---|---|---|
| 1 | 67.1 / 67.2 | 52.7 / 52.8 | 0.09 / 0.13 s |
| 2 | 120.8 / 61.1 | 93.6 / 47.3 | 0.19 / 0.26 s |
| 3 | 135.7 / 46.3 | 111.1 / 37.8 | 0.28 / 0.39 s |
| 4 | 204.0 / 52.5 | 164.9 / 42.5 | 0.37 / 0.52 s |
