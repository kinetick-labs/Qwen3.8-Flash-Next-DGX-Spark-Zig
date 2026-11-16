# Kinetick-labs TensorFold runtime image for one DGX Spark (GB10, aarch64, CUDA 13): what serve.sh runs
# tensorfold-native in. The engine binary is not in the image; scripts/build.sh builds it and serve.sh mounts it.
#
#   NVIDIA PyTorch 26.07 (CUDA 13 libraries, torch, Triton 3.7.1, Python 3.12)
#   + /opt/tensorfold/python/tensorfold   the Python package --vision-helper runs (tensorfold.vision.native_helper):
#                                         upstream ashhart/TensorFold 4af95036e (the last tree with the Python engine,
#                                         before 1.0.0) + python-overlay.patch (vision helper, FP8 KV and Flash Next
#                                         kernel sources)
#   + /opt/tensorfold/share/tensorfold/cuda/sm121   the captured Triton kernel set the mia engine replays
#                                         (aot.json + cubins/), compiled here for sm_121 without a GPU by
#                                         tools/zig/flashnext_aot.py from the pinned TensorFold spec, and checked
#                                         cubin by cubin against sm121-kernels.txt
#   + transformers 5.17.0 and PyAV 19.0.1 (the vision helper's processor and video decoder)
#
# Build with scripts/build-image.sh (no GPU needed). Checked against our local tensorfold-zig:exl3-572414d: same kernel
# set (all 320 cubins byte-identical), same pip packages, same libraries; the Python tree differs by one comment line.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c

FROM ${BASE_IMAGE} AS kernels
ARG TF_REPO=https://github.com/kinetick-labs/TensorFold
# the Python tree: upstream 4af95036e ("lean_check: ..."), parent of 394d566cb "TensorFold 1.0.0 is the Zig engine"
ARG TF_PY_REF=4af95036e59e86188b381bdc66ad3a0be664ad15
# the kernel spec and its compiler: kinetick-labs/TensorFold main 082aee98a
ARG TF_AOT_REF=082aee98a4e9247a9a3732291fba1b8884e14416
ARG JOBS=8
COPY python-overlay.patch python-mtimes.txt sm121-kernels.txt kernel-set.py /ctx/
RUN set -eu; \
    fetch() { rm -rf "$2"; git init -q "$2"; git -C "$2" fetch -q --depth 1 "$TF_REPO" "$1"; git -C "$2" checkout -q FETCH_HEAD; }; \
    fetch "$TF_PY_REF" /src/py; fetch "$TF_AOT_REF" /src/aot; \
    mkdir -p /tensorfold/src; cp -a /src/py/src/tensorfold /tensorfold/src/tensorfold; \
    patch -s -p1 -d /tensorfold/src < /ctx/python-overlay.patch
# Triton's line tables carry each source file's mtime: set those of the tree our kernel set was first compiled from
RUN set -eu; find /tensorfold/src/tensorfold -exec touch -h -d @1791328680 {} +; \
    while read -r t f; do touch -d "@$t" "/tensorfold/src/tensorfold/$f"; done < /ctx/python-mtimes.txt
# the spec's TP=1 entries compiled for sm_121 (no GPU) with the package at the spec's source_root; the tool reports
# 18 entries captured from another Python tree, and kernel-set.py check keeps the listed set only
RUN set -eu; cd /src/aot; \
    python3 /ctx/kernel-set.py filter zig/tests/cuda/flashnext/kernels.json /src/spec.json; \
    PYTHONPATH=/tensorfold/src python3 tools/zig/flashnext_aot.py build --spec /src/spec.json \
      --jit zig/tests/cuda/flashnext/jit.json --tp 1 --jobs "$JOBS" --out /out/sm121 || true; \
    python3 /ctx/kernel-set.py check /out/sm121

FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="Kinetick-labs TensorFold runtime (DGX Spark, CUDA 13)" \
      tf.python.ref="4af95036e59e86188b381bdc66ad3a0be664ad15+python-overlay.patch" \
      tf.aot.ref="082aee98a4e9247a9a3732291fba1b8884e14416"
COPY --from=kernels /tensorfold/src/tensorfold /opt/tensorfold/python/tensorfold
COPY --from=kernels /out/sm121 /opt/tensorfold/share/tensorfold/cuda/sm121
RUN pip install --no-cache-dir --no-deps "transformers==5.17.0" "av==19.0.1" \
    && PYTHONPATH=/opt/tensorfold/python python3 -c "import tensorfold.vision.native_helper, transformers, av, PIL; print('[image] vision helper imports')"
ENV HF_HOME=/root/.cache/huggingface \
    TENSORFOLD_CUDA_KERNELS=/opt/tensorfold/share/tensorfold/cuda/sm121 \
    CUDA_CACHE_PATH=/cache/nv \
    TENSORFOLD_VISION_PYTHONPATH=/opt/tensorfold/python
WORKDIR /workspace
