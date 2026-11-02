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
