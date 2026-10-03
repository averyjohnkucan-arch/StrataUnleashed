"""Run CUDA numerical checks; optional --model adds real expert and PLE checks.
All logs and temporary files stay in this checkout's work directory.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from unleashed import configure_environment, split_paths
from tools.gguf_reader import GGUFFile


def run(label, args, extra=None):
    executable = ROOT / "build" / (args[0] + (".exe" if os.name == "nt" else ""))
    with (ROOT / "work" / f"{label}.log").open("w") as out:
        result = subprocess.run(
            [str(executable), *map(str, args[1:])],
            cwd=ROOT,
            stdout=out,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, **(extra or {})),
        )
    print(label, result.returncode, flush=True)
    return {"test": label, "code": result.returncode}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model", type=Path, help="Any shard of a qwen4exp model for read-only checks"
    )
    a = ap.parse_args()
    configure_environment()
    if os.name != "nt":
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    results = [
        run("parity-mixed", ["kv_mixed_parity"]),
        run(
            "parity-formats",
            [
                "native_expert_parity",
                "--synthetic",
                "q6_K/q8_0",
                "q8_0/q8_0",
                "q4_K/q5_1",
                "q5_K/q8_0",
                "q4_0/q4_0",
                "q4_1/q5_0",
                "nvfp4/nvfp4",
                "mxfp4/mxfp4",
            ],
        ),
        run(
            "parity-large",
            [
                "native_expert_parity",
                "--synthetic",
                "q4_K/q4_K",
                "q5_K/q5_K",
                "q6_K/q6_K",
                "q8_0/q6_K",
            ],
            {"STRATA_PARITY_FF": "768"},
        ),
    ]
    if a.model:
        shards = split_paths(a.model.resolve())
        layers = int(GGUFFile(shards[0]).metadata["qwen4exp.block_count"])
        probes = sorted({v for v in (0, 2, 11, 20, 41, layers - 1) if v < layers})
        results.append(
            run("parity-model-experts", ["native_expert_parity", shards[0], *probes])
        )
        ple = next(
            (
                p
                for p in shards
                if any(
                    t.name == "per_layer_token_embd.weight" for t in GGUFFile(p).tensors
                )
            ),
            None,
        )
        if ple:
            results.append(run("parity-model-ple", ["ple_q8_parity", ple, "direct"]))
    (ROOT / "work/test-results.json").write_text(json.dumps(results, indent=2) + "\n")
    return int(any(r["code"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
