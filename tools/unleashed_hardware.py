"""Machine inspection and conservative fit estimates for resident-expert inference."""

from __future__ import annotations
import csv
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
import psutil
import unleashed
from tools.unleashed_catalog import ROOT
from tools.unleashed_download import remaining_download
from tools.unleashed_storage import scan_storage

from tools.unleashed_policy import NATIVE_CONTEXT, default_kv

GIB = 1024**3
MIB = 1024**2


def scan_system():
    ram = psutil.virtual_memory()
    errors, cards = [], []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        for row in csv.reader(result.stdout.splitlines()):
            idx, name, total, free, cc, driver = [x.strip() for x in row]
            cards.append(
                {
                    "index": int(idx),
                    "name": name,
                    "total_mib": float(total),
                    "free_mib": float(free),
                    "compute_capability": float(cc),
                    "driver": driver,
                }
            )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        errors.append(
            f"NVIDIA GPU/driver scan unavailable ({type(exc).__name__}); this tuner requires one NVIDIA CUDA GPU."
        )
    cpu = unleashed.cpu_identity()
    cpu["avx2"] = None
    if platform.system() == "Linux" and Path("/proc/cpuinfo").exists():
        cpu["avx2"] = "avx2" in Path("/proc/cpuinfo").read_text().split()
    tools = {
        name: shutil.which(name)
        for name in ("cmake", "ninja", "nvcc", "cl" if os.name == "nt" else "c++")
    }
    cuda = (
        Path(os.environ.get("CUDA_PATH", "/usr/local/cuda"))
        / "bin"
        / ("nvcc.exe" if os.name == "nt" else "nvcc")
    )
    if not tools["nvcc"] and cuda.is_file():
        tools["nvcc"] = str(cuda)
    if not tools["nvcc"] and os.name != "nt":
        tools["nvcc"] = next(
            (
                str(p)
                for p in sorted(
                    Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True
                )
            ),
            None,
        )
    engine_runs = False
    if unleashed.engine_path().is_file():
        try:
            check = subprocess.run(
                [str(unleashed.engine_path()), "--help"],
                capture_output=True,
                timeout=15,
            )
            engine_runs = check.returncode == 0
        except (OSError, subprocess.SubprocessError):
            pass
    built_for = ""
    try:
        built_for = json.loads((ROOT / "build/UNLEASHED-BUILD.json").read_text())["gpu"]
    except (OSError, KeyError, ValueError):
        pass
    packaged_architectures = []
    try:
        manifest = json.loads((ROOT / "MANIFEST.json").read_text())
        packaged_architectures = manifest.get("requirements", {}).get(
            "cuda_architectures", []
        )
    except (OSError, ValueError):
        pass
    storage = scan_storage(
        ROOT, unleashed.engine_path() if unleashed.engine_path().is_file() else None
    )
    return {
        "os": platform.system(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "ram_total": ram.total,
        "ram_available": ram.available,
        "disk_free": shutil.disk_usage(ROOT).free,
        "storage_path": str(ROOT / "models"),
        "gpus": cards,
        "tools": tools,
        "errors": errors,
        "engine_runs": engine_runs,
        "engine_built_for": built_for,
        "engine_architectures": packaged_architectures,
        "storage": storage,
    }


def assess(
    entry,
    system,
    gpu_index=0,
    reserve_mib=0,
    context=NATIVE_CONTEXT,
    download_bytes=None,
    kv=None,
):
    """Conservative estimates for this tuner's resident-expert mode; never promise a fit."""
    ins = entry["inspection"]
    blocked = list(ins["reasons"])
    warnings = []
    if system["os"] not in ("Linux", "Windows") or system["machine"].lower() not in (
        "amd64",
        "x86_64",
    ):
        blocked.append("This CLI supports x86-64 Windows and Linux")
    if system["cpu"].get("avx2") is False:
        blocked.append("The CPU lacks AVX2, required by this build")
    gpu = next((g for g in system["gpus"] if g["index"] == gpu_index), None)
    if gpu is None:
        blocked.append(
            "No usable NVIDIA GPU selected (AMD/CPU-only tuning is not implemented)"
        )
    elif gpu["compute_capability"] < 7.5:
        blocked.append(
            "GPU compute capability must be at least 7.5 (RTX 20 series or newer)"
        )
    # The PLE table is mapped from disk, not counted as a permanently resident RAM allocation.
    # Include expert arena, two dense copies, OS/loader overhead, and modest context growth.
    ram_need = (
        ins["expert_bytes"]
        + 2 * ins["dense_bytes"]
        + 6 * GIB
        + max(0, context - 16384) * 65536
    )
    disk_need = (
        (remaining_download(entry) if download_bytes is None else download_bytes)
        + ins["pack_bytes"]
        + 2 * GIB
    )
    # The supported architecture has 12 full-attention layers, two KV heads,
    # and 256 values per head. Other layers use recurrent state.
    pair = kv or default_kv(gpu["total_mib"]) if gpu else (kv or "FP16/FP16")
    bits = [16 if part == "FP16" else int(part[1:]) for part in pair.split("/")]
    row_bytes = sum(512 if b == 16 else 8 * (2 + 4 * b) for b in bits)
    vram_need = ins["dense_bytes"] + context * 12 * 2 * row_bytes + 2 * GIB
    if ram_need > system["ram_total"]:
        blocked.append(
            f'Estimated resident RAM need {ram_need / GIB:.1f} GiB exceeds installed {system["ram_total"] / GIB:.1f} GiB'
        )
    elif ram_need > system["ram_available"]:
        blocked.append(
            f'Estimated RAM need {ram_need / GIB:.1f} GiB exceeds currently available {system["ram_available"] / GIB:.1f} GiB; close other apps'
        )
    if disk_need > system["disk_free"]:
        blocked.append(
            f'Need about {disk_need / GIB:.1f} GiB more disk space; {system["disk_free"] / GIB:.1f} GiB free'
        )
    budget = 0
    if gpu:
        safety = max(512, int(gpu["total_mib"] * 0.02 + 0.999))
        budget = (gpu["free_mib"] - reserve_mib - safety) * MIB
        if budget < vram_need:
            blocked.append(
                f"Estimated startup VRAM {vram_need / GIB:.1f} GiB exceeds {max(0,budget) / GIB:.1f} GiB available after reservation/headroom"
            )
    needs_build = not system.get("engine_runs", False)
    if gpu and system.get("engine_runs"):
        local_build = gpu["name"] in system.get("engine_built_for", "")
        bundled_build = round(gpu["compute_capability"] * 10) in system.get(
            "engine_architectures", []
        ) and not system.get("engine_built_for")
        needs_build = not (local_build or bundled_build)
    missing = [n for n, p in system["tools"].items() if not p]
    if needs_build:
        if missing:
            blocked.append(
                "No compatible engine ready; source build needs: "
                + ", ".join(missing)
                + (
                    " (use an x64 Visual Studio developer terminal)"
                    if system["os"] == "Windows"
                    else ""
                )
            )
        else:
            warnings.append(
                "The engine will be built locally for this machine before tuning"
            )
    warnings.append(
        "RAM/VRAM are conservative estimates; measured tuning decides the actual fit and speed"
    )
    return {
        "status": "cannot-run-now" if blocked else "candidate",
        "reasons": blocked,
        "warnings": warnings,
        "ram_estimate_bytes": ram_need,
        "additional_disk_bytes": disk_need,
        "kv": pair,
        "context_tokens": context,
        "vram_estimate_bytes": vram_need,
        "vram_budget_bytes": max(0, budget),
        "needs_build": needs_build,
    }


def recommendations(rows, intent):
    eligible = [r for r in rows if r["assessment"]["status"] == "candidate"]
    preferred = {
        "small": ["ista", "ista-coder"],
        "coder": ["ista-coder", "ista"],
        "large": ["atomic", "ista"],
        "uncensored": ["huihui"],
    }[intent]
    if intent == "uncensored":
        eligible = [r for r in eligible if r["model"]["provider"] == "huihui"]

    def rank(row):
        e = row["model"]
        provider = e["provider"]
        priority = (
            preferred.index(provider) if provider in preferred else len(preferred)
        )
        size = (
            e["download_bytes"]
            if intent in ("small", "coder", "uncensored")
            else -e["download_bytes"]
        )
        return priority, size

    return sorted(eligible, key=rank)
