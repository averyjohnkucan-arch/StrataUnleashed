#!/usr/bin/env python3
"""Strata Unleashed: prepare a GGUF model, self-tune once per machine/model, serve locally."""

from __future__ import annotations
import argparse, hashlib, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]


def configure_environment():
    for name in ("tmp", "cache", "config"):
        (ROOT / "work" / name).mkdir(parents=True, exist_ok=True)
    for key, val in {
        "TMPDIR": "tmp",
        "TEMP": "tmp",
        "TMP": "tmp",
        "CUDA_CACHE_PATH": "cache",
        "XDG_CACHE_HOME": "cache",
        "PIP_CACHE_DIR": "cache",
        "XDG_CONFIG_HOME": "config",
    }.items():
        os.environ[key] = str(ROOT / "work" / val)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["STRATA_GGUF_PY"] = str(ROOT / "third_party/llama.cpp/gguf-py")


def run(args):
    subprocess.run([str(x) for x in args], cwd=ROOT, check=True)


def engine_path():
    return ROOT / "build" / ("strata.exe" if os.name == "nt" else "strata")


def build():
    cmake = shutil.which("cmake")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        candidates = [
            Path(os.environ.get("CUDA_PATH", "/usr/local/cuda"))
            / "bin"
            / ("nvcc.exe" if os.name == "nt" else "nvcc")
        ]
        if os.name != "nt":
            candidates += sorted(
                Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True
            )
        nvcc = next((str(p) for p in candidates if p.is_file()), None)
    if not cmake or not nvcc:
        raise RuntimeError(
            "Install CMake, Ninja, a C++ compiler, and CUDA Toolkit; on Windows run from an x64 Visual Studio developer terminal."
        )
    args = [
        cmake,
        "-S",
        ROOT,
        "-B",
        ROOT / "build",
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DSTRATA_ENABLE_CUDA=ON",
        "-DSTRATA_PORTABLE=ON",
        "-DSTRATA_BUILD_TESTS=OFF",
        "-DSTRATA_MMQ_KQUANTS=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=native",
        f"-DCMAKE_CUDA_COMPILER={nvcc}",
        f'-DSTRATA_GGML_DIR={ROOT / "third_party/llama.cpp"}',
    ]
    run(args)
    run(
        [
            cmake,
            "--build",
            ROOT / "build",
            "--target",
            "strata",
            "--parallel",
            min(os.cpu_count() or 4, 12),
        ]
    )
    (ROOT / "build/UNLEASHED-BUILD.json").write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "gpu": gpu_identity(),
            },
            indent=2,
        )
        + "\n"
    )


def gpu_identity():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def cpu_identity():
    identity = {"processor": platform.processor(), "logical_cpus": os.cpu_count()}
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                identity["model"] = winreg.QueryValueEx(key, "ProcessorNameString")[
                    0
                ].strip()
        except OSError:
            pass
    else:
        path = Path("/proc/cpuinfo")
        if path.exists():
            fields = dict(
                line.split(":", 1)
                for line in path.read_text().splitlines()
                if ":" in line
            )
            identity["model"] = next(
                (v.strip() for k, v in fields.items() if k.strip() == "model name"), ""
            )
    try:
        import psutil

        identity["physical_cores"] = psutil.cpu_count(logical=False)
        identity["affinity"] = sorted(psutil.Process().cpu_affinity())
    except (ImportError, AttributeError, OSError):
        pass
    return identity


def signature(cfg):
    from tools.gguf_reader import GGUFFile

    model = Path(cfg["args"][cfg["args"].index("--native") + 1])
    shards = split_paths(model)
    evidence = {
        "schema": 3,
        "tuning": cfg.get("unleashed_tuning", {}),
        "system": platform.platform(),
        "cpu": cpu_identity(),
        "cpu_count": os.cpu_count(),
        "gpu": gpu_identity(),
        "args": cfg["args"],
        "gpu_selection": cfg.get("gpu", 0),
        "lib_dirs": cfg.get("lib_dirs", []),
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("STRATA_")
            or k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")
        },
        "tune_mtp": cfg.get("tune_mtp"),
        "weights": [
            (str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in shards
        ],
    }
    for key, p in [
        ("engine", Path(cfg["exe"])),
        ("tuner", ROOT / "tools/unleashed_tune.py"),
        ("server", ROOT / "serve/server.py"),
    ]:
        evidence[key] = hashlib.sha256(p.read_bytes()).hexdigest()
    if "--expert-profile" in cfg["args"]:
        profile = Path(cfg["args"][cfg["args"].index("--expert-profile") + 1])
        evidence["expert_profile_sha256"] = hashlib.sha256(
            profile.read_bytes()
        ).hexdigest()
    if cfg.get("tune_mtp"):
        evidence["mtp_files"] = [
            (str(p), p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(Path(cfg["tune_mtp"]).glob("*"))
            if p.is_file()
        ]
    pack = Path(cfg["args"][cfg["args"].index("--pack") + 1])
    evidence["pack"] = [
        (str(p), p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(pack.glob("*"))
        if p.is_file()
    ]
    return (
        hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest(),
        evidence,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model", type=Path, help="Any shard of a complete qwen4exp GGUF model"
    )
    ap.add_argument(
        "--mtp", type=Path, help="Optional prepared MTP runtime; absence is supported"
    )
    ap.add_argument(
        "--config", type=Path, help="Existing engine config instead of --model"
    )
    ap.add_argument("--context", type=int, default=16384)
    ap.add_argument(
        "--port", type=int, help="HTTP port; overrides the config (default 8100)"
    )
    ap.add_argument(
        "--gpu", type=int, help="Physical NVIDIA GPU index for single-GPU tuning"
    )
    ap.add_argument(
        "--reserve-vram-mib",
        type=int,
        default=0,
        help="Extra VRAM kept free for other apps, in MiB; automatic safety margin is separate",
    )
    ap.add_argument(
        "--retune",
        action="store_true",
        help="Discard the saved tuning decision and measure again",
    )
    ap.add_argument("--tune-only", action="store_true")
    ap.add_argument(
        "--chat",
        action="store_true",
        help="Open terminal test chat with no system prompt",
    )
    ap.add_argument("--chat-prompt", help="With --chat, run one prompt and exit")
    ap.add_argument(
        "--build", action="store_true", help="Build the engine on this machine"
    )
    a = ap.parse_args()
    configure_environment()
    if a.reserve_vram_mib < 0:
        ap.error("--reserve-vram-mib must not be negative")
    if a.gpu is not None and a.gpu < 0:
        ap.error("--gpu must not be negative")
    if a.build:
        build()
    if not a.model and not a.config:
        ap.error("--model or --config is required")
    if a.config:
        cfg = json.loads(a.config.read_text())
        cfg["exe"] = str(Path(cfg["exe"]).resolve())
        if not Path(cfg["exe"]).is_file():
            raise FileNotFoundError(cfg["exe"])
    else:
        if not engine_path().is_file():
            build()
        if a.context < 9216:
            ap.error(
                "--context must be at least 9216 for the 8192/512 prefill validation workload"
            )
        model = split_paths(a.model.resolve())[0]
        ident = hashlib.sha256(str(model).encode()).hexdigest()[:16]
        pack = ROOT / "work/packs" / ident
        # Conversion writes a completion receipt only after all artifacts are present.
        stamp = {
            "packer": hashlib.sha256(
                (ROOT / "tools/iq_pack.py").read_bytes()
            ).hexdigest(),
            "shards": [
                (str(p), p.stat().st_size, p.stat().st_mtime_ns)
                for p in split_paths(model)
            ],
        }
        receipt = pack / "unleashed-model.json"
        if (
            not receipt.exists()
            or not all(
                (pack / p).is_file()
                for p in (
                    "index.txt",
                    "dense.bin",
                    "native_experts.txt",
                    "tokenizer/vocab.json",
                )
            )
            or json.loads(receipt.read_text()) != json.loads(json.dumps(stamp))
        ):
            pack.mkdir(parents=True, exist_ok=True)
            run(
                [
                    sys.executable,
                    "-B",
                    ROOT / "tools/iq_pack.py",
                    "--gguf",
                    model,
                    "--out",
                    pack,
                    "--compat-bf16",
                ]
            )
            receipt.write_text(json.dumps(stamp, indent=2) + "\n")
        from tools.gguf_reader import GGUFFile

        g = GGUFFile(model)
        n = int(g.metadata["qwen4exp.block_count"])
        e = int(g.metadata["qwen4exp.expert_count"])
        profile = ROOT / "data/expert-profile.bin"
        if not profile_matches(profile, n, e):
            # Pruned variants cannot reuse a profile with a different expert count.
            profile = ROOT / "work" / f"profile-{ident}.bin"
            if not profile_matches(profile, n, e):
                make_profile(profile, n, e)
        args = [
            "--pack",
            str(pack),
            "--native",
            str(model),
            "--expert-profile",
            str(profile),
            "--expert-cache",
            "auto",
            "--max-context",
            str(a.context),
            "--kv",
            "fp16",
            "--pool-affinity",
            "p-cores",
            "--prompt-cache",
            "0",
            "--ple-io",
            "mmap",
            "--spec",
            "2",
            "--suffix-draft",
            "0",
            "--prefill",
            "512",
        ]
        cfg = {
            "exe": str(engine_path()),
            "cwd": str(ROOT),
            "args": args,
            "tokenizer": str(pack / "tokenizer"),
            "model_name": "StrataUnleashed-" + model.stem,
            "port": a.port or 8100,
            "log": str(ROOT / "work/engine.log"),
            "gpu": 0,
        }
        if a.mtp:
            cfg["tune_mtp"] = str(a.mtp.resolve())
    if a.gpu is not None:
        cfg["gpu"] = a.gpu
    cfg["unleashed_tuning"] = {"reserve_vram_mib": a.reserve_vram_mib}
    key, evidence = signature(cfg)
    out = ROOT / "work/autotune" / key
    out.mkdir(parents=True, exist_ok=True)
    source = out / "input-config.json"
    source.write_text(json.dumps(cfg, indent=2) + "\n")
    (out / "fingerprint.json").write_text(json.dumps(evidence, indent=2) + "\n")
    best = out / "best-config.json"
    report = out / "RESULTS.json"
    if a.retune or not (best.exists() and report.exists()):
        print(
            f"Self-tuning for maximum practical GPU residency, reserving {a.reserve_vram_mib} MiB for other apps plus automatic safety headroom; 512/512 decode, then prefill within 10% decode loss.",
            flush=True,
        )
        run(
            [
                sys.executable,
                "-B",
                ROOT / "tools/unleashed_tune.py",
                source,
                "--out",
                out,
                "--reserve-vram-mib",
                a.reserve_vram_mib,
            ]
        )
    if a.tune_only:
        print(f"Validated configuration: {best}")
        return 0
    port = a.port if a.port is not None else cfg.get("port", 8100)
    if a.chat:
        from tools.unleashed_chat import session

        return session(best, prompt=a.chat_prompt)
    print(
        f"Starting Strata Unleashed at http://127.0.0.1:{port} using {best}",
        flush=True,
    )
    run(
        [
            sys.executable,
            "-B",
            "-m",
            "serve.server",
            "--engine",
            "strata",
            "--config",
            best,
            "--port",
            str(port),
        ]
    )
    return 0


def split_paths(first):
    import re

    first = Path(first)
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first.name)
    paths = (
        [first]
        if not m
        else [
            first.with_name(
                first.name[: m.start()] + f"-{i:05}-of-{int(m.group(2)):05}.gguf"
            )
            for i in range(1, int(m.group(2)) + 1)
        ]
    )
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing shards: " + ", ".join(missing))
    return paths


def profile_matches(path, layers, experts):
    import struct

    try:
        with Path(path).open("rb") as stream:
            magic, version, nl, ne, pairs, ranks = struct.unpack(
                "<4s5I", stream.read(24)
            )
        return (
            magic == b"STRP"
            and version == 1
            and (nl, ne) == (layers, experts)
            and pairs <= layers * experts
            and ranks == layers * experts
            and Path(path).stat().st_size == 24 + 4 * pairs + 4 * ranks
        )
    except (OSError, struct.error):
        return False


def make_profile(path, layers, experts):
    import struct

    pairs = [(l, e) for e in range(experts) for l in range(layers)]
    rank = {pair: r for r, pair in enumerate(pairs)}
    data = b"STRP" + struct.pack("<5I", 1, layers, experts, len(pairs), len(pairs))
    data += b"".join(struct.pack("<2H", *pair) for pair in pairs)
    data += b"".join(
        struct.pack("<i", rank[l, e]) for l in range(layers) for e in range(experts)
    )
    Path(path).write_bytes(data)


if __name__ == "__main__":
    raise SystemExit(main())
