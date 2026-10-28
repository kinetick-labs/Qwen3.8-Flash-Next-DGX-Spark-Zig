# Only two patches, and they are commits

This directory is deliberately empty of `*.patch`. The two patches this fork needs are **commits** on the
revision `scripts/config.sh` pins:

    TF_REPO=kinetick-labs/TensorFold
    TF_REF=1000c39c169a3f973464966d9999409b0e3b4b96   # integration/1.1.7-vision

That branch is TensorFold v1.1.7 (`c5953415e`) plus two of MiaAI-Lab's seventeen recipe patches:

- `0001` — the NVIDIA FP4 checkpoint's quantized vision tower decoded to bf16 at load;
- `0002` — the per-request image-token budget at 16384.

Its tree is byte-identical to a pristine v1.1.7 checkout after `git apply` of those two files (both trees
hash `38165d03c8`). Leaving the patch files here would make the image build apply them a second time — the
zero-delta overlay that reads as the engine's provenance while carrying nothing. `scripts/prepare.sh` skips
this directory when it holds no `*.patch`, and the image's `tf.patches` label hashes `TF_REPO`/`TF_REF` as
well, so a pin change still forces a rebuild.

The series' other fifteen patches are **not** carried: upstream 1.1.5 and 1.1.7 ship the prompt path and the
n-gram reader they were written for, and the remainder (fp8 KV, sampled-draft batching, the keep-late log,
the shared side stream, the stop rule) needs a port onto 1.1.7's engine and none of it is in this branch.
`docs/audit-v1.1.7-rebase.md` gives the one-line verdict and the evidence for each.

Patches' origin: [MiaAI-Lab's
recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) (MIT), `patches/0001`
and `0002`; each commit on the integration branch names the file it came from.
