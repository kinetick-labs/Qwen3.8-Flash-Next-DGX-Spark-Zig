# Empty on purpose

MiaAI-Lab's seventeen recipe patches (`0001` to `0017` against TensorFold v1.1.4) are **commits** on the
revision `scripts/config.sh` pins:

    TF_REPO=kinetick-labs/TensorFold
    TF_REF=fa6f9e18f529d416ca60d53d2ea17a8af483ae2b   # integration/1.1.4-mia17

That branch is v1.1.4 (`eab85828`) plus the series, one commit per patch, and its tree is byte-identical
to a pristine v1.1.4 checkout after `git apply patches/*.patch` in filename order. Leaving the patch files
here would make the image build apply them a second time — the zero-delta overlay that reads as the
engine's provenance while carrying nothing. `scripts/prepare.sh` skips this directory when it holds no
`*.patch`, and the image's `tf.patches` label hashes `TF_REPO`/`TF_REF` as well, so a pin change still
forces a rebuild.

Where the patches come from: [MiaAI-Lab's
recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) (MIT), `patches/0001`
to `0017`; each commit on the integration branch names the patch file it came from. What each one does,
and which of our own recipe's fixes it retires, is in `docs/audit-v1.1.4-rebase.md`.
