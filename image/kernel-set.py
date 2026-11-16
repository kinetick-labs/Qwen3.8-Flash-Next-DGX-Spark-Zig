#!/usr/bin/env python3
"""Our sm_121 Triton kernel set from the pinned TensorFold spec (used by the Dockerfile).

  filter SPEC OUT    the spec's TP=1 entries, without the capture's cubin sha256 (sm121-kernels.txt is the reference)
  check  SET_DIR     drop the built kernels sm121-kernels.txt does not list (18 spec entries captured from another
                     Python tree: EXL3 _gemm/_f16_mm/_ple_rows and four _hc_reduce_act variants, never in our
                     runtime), then require exactly the listed hashes and cubin sha256s
"""
import hashlib
import json
import pathlib
import sys

LIST = pathlib.Path(__file__).with_name("sm121-kernels.txt")


def listed() -> dict:
    return dict(line.split()[:2] for line in LIST.read_text().splitlines() if line.strip())


def filter_spec(src: str, out: str) -> None:
    spec = json.loads(pathlib.Path(src).read_text())
    assert spec.get("source_root") == "/tensorfold/src", spec.get("source_root")
    spec["kernels"] = [{k: v for k, v in e.items() if k != "cubin_sha256"} for e in spec["kernels"]
                       if 1 in e.get("tp", [1])]
    pathlib.Path(out).write_text(json.dumps(spec))
    print(f"[kernels] {len(spec['kernels'])} TP=1 spec entries")


def check(set_dir: str) -> None:
    d = pathlib.Path(set_dir)
    aot = json.loads((d / "aot.json").read_text())
    want = listed()
    extra = [k for k in aot["kernels"] if k["hash"] not in want]
    for k in extra:
        (d / "cubins" / f"{k['hash']}.cubin").unlink(missing_ok=True)
    aot["kernels"] = [k for k in aot["kernels"] if k["hash"] in want]
    (d / "aot.json").write_text(json.dumps(aot, indent=1) + "\n")
    print(f"[kernels] dropped {len(extra)} unlisted: {sorted({k['fn'] for k in extra})}")
    got = {k["hash"]: k for k in aot["kernels"]}
    assert set(got) == set(want), f"kernel set differs from the list: {len(set(got) ^ set(want))} hashes"
    bad = [h for h, s in want.items() if hashlib.sha256((d / "cubins" / f"{h}.cubin").read_bytes()).hexdigest() != s]
    assert not bad, f"{len(bad)} cubins differ from the list, e.g. {got[bad[0]]['fn']} {bad[0][:12]}"
    print(f"[kernels] {len(got)} kernels, every cubin equal to sm121-kernels.txt")


if __name__ == "__main__":
    {"filter": filter_spec, "check": check}[sys.argv[1]](*sys.argv[2:])
