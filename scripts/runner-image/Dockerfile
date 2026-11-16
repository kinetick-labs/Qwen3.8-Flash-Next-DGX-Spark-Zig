# Runtime image for the compute lane's in-model runner (Jay Leaton, jayleaton/qwen4-tensorfold-spark, Apache-2.0):
# the runtime image (scripts/build-image.sh) plus clang-18/lld-18 (the runner compiles its cells) and
# runner-assets-20261107 (calc.tfpc and the graph-v7-powmod bank). Build with scripts/runner-image/build.sh.
ARG BASE_IMAGE=tensorfold-zig:runtime-pt2607
FROM ${BASE_IMAGE}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends clang-18 lld-18 \
    && ln -sf clang-18 /usr/bin/clang && ln -sf ld.lld-18 /usr/bin/ld.lld && rm -rf /var/lib/apt/lists/*
COPY runner/calc.tfpc /opt/tensorfold/share/runner/calc.tfpc
COPY runner/graph-v7-powmod/bank/ /opt/tensorfold/share/runner/graph-v7-powmod/bank/
COPY runner/graph-v7-powmod/bank/Qwenputer-MIT.txt /opt/tensorfold/licenses/runtime-deps/Qwenputer-MIT.txt
