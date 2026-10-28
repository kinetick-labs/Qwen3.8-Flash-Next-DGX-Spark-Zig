# Only two patches, and they are commits

This directory is deliberately empty of `*.patch`. The two patches this fork needs are **commits** on the
revision `scripts/config.sh` pins:

    TF_REPO=kinetick-labs/TensorFold
    TF_REF=ba0f4941a35719a2824150dc31d35a2e3f673058   # integration/1.1.8-vision

That branch is TensorFold v1.1.8 (`03d08f1ef`) plus two of MiaAI-Lab's seventeen recipe patches:

- `0001` — the NVIDIA FP4 checkpoint's quantized vision tower decoded to bf16 at load;
- `0002` — the cap on one request's image tokens at 16384 (the default stays 4096, so
  `--vision-image-tokens 16384` is accepted instead of refused).

Its tree is byte-identical to a pristine v1.1.8 checkout after `git apply` of those two files (both trees
hash `80004eceb6`). Leaving the patch files here would make the image build apply them a second time — the
zero-delta overlay that reads as the engine's provenance while carrying nothing. `scripts/prepare.sh` skips
this directory when it holds no `*.patch`, and the image's `tf.patches` label hashes `TF_REPO`/`TF_REF` as
well, so a pin change still forces a rebuild.

The series' other fifteen patches are **not** carried: upstream 1.1.5 and 1.1.7 ship the prompt path and the
n-gram reader they were written for, and the remainder (fp8 KV, sampled-draft batching, the keep-late log,
the shared side stream, the stop rule) needs a port onto the engine and none of it is in this branch.
`docs/audit-upstream-rebase.md` gives the one-line verdict and the evidence for each.

Patches' origin: [MiaAI-Lab's
recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) (MIT), `patches/0001`
and `0002`; each commit on the integration branch names the file it came from.
