"""Interactive model selection, hardware advice, verified downloads and procedural tuning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import psutil
import requests
from tqdm import tqdm

import unleashed
from tools.unleashed_catalog import (
    CATALOG,
    ROOT,
    SOURCES,
    inspect_headers,
    resolve_url,
    safe_path,
    refresh_catalog,
)
from tools.gguf_reader import GGUFFile

GIB = 1024**3
MIB = 1024**2


@contextmanager
def download_lock():
    """Protect resumable files from a second setup process on either platform."""
    with (ROOT / "work/model-download.lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt

                lock.seek(0)
                if not lock.read(1):
                    lock.write(b"0")
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError(
                "Another Unleashed model download is active; wait for it to finish"
            ) from exc
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def version():
    return re.search(
        r"project\(strata VERSION ([0-9.]+)",
        (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig"),
    )[1]


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
    }


def model_directory(entry):
    # Repository ID and commit isolate providers, revisions, and similarly named quants.
    provider = entry["provider"]
    quant = re.sub(r"[^A-Za-z0-9_.-]", "_", entry["quant"])
    return ROOT / "models" / provider / quant / entry["revision"][:12]


def remaining_download(entry):
    dest = model_directory(entry)
    remaining = 0
    for f in entry["files"]:
        path = dest / Path(f["path"]).name
        part = path.with_suffix(path.suffix + ".part")
        size = (
            path.stat().st_size
            if path.is_file()
            else (part.stat().st_size if part.is_file() else 0)
        )
        remaining += max(0, f["size"] - min(size, f["size"]))
    return remaining


def assess(
    entry, system, gpu_index=0, reserve_mib=0, context=16384, download_bytes=None
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
    # Upper bound for 48 layers of FP16 K/V at 2 heads x 256 values + index/workspace allowance.
    vram_need = ins["dense_bytes"] + context * 48 * 2 * 256 * 4 + 2 * GIB
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
        if gpu["compute_capability"] != 8.9 and (ROOT / "build/strata").exists():
            warnings.append(
                "The bundled Linux binary targets Ada (sm89); build locally for this GPU"
            )
    needs_build = not system.get("engine_runs", False)
    if gpu and system.get("engine_runs"):
        local_build = gpu["name"] in system.get("engine_built_for", "")
        bundled_build = (
            system["os"] == "Linux"
            and gpu["compute_capability"] == 8.9
            and not system.get("engine_built_for")
        )
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


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * MIB), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url, dest, size, digest, session=requests, progress=True):
    """Resume only the pinned object; validate range, length and SHA256 before rename."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        if dest.stat().st_size == size and sha256(dest) == digest:
            return dest
        raise ValueError(
            f"Existing file failed verification: {dest}; move it aside before retrying"
        )
    part = dest.with_suffix(dest.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        raise ValueError(f"Partial download exceeds expected length: {part}")
    if offset < size:
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with session.get(url, headers=headers, stream=True, timeout=(15, 60)) as r:
            r.raise_for_status()
            if offset and r.status_code == 200:
                offset = 0  # Server ignored Range; restart instead of appending corrupt data.
            elif r.status_code == 206:
                m = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)", r.headers.get("Content-Range", "")
                )
                if not m or int(m[1]) != offset or int(m[3]) != size:
                    raise ValueError("Download server returned an invalid byte range")
            elif r.status_code != 200:
                raise ValueError(f"Unexpected download status: {r.status_code}")
            with part.open("ab" if offset else "wb") as out, tqdm(
                total=size,
                initial=offset,
                unit="B",
                unit_scale=True,
                desc=dest.name,
                disable=not progress,
            ) as bar:
                for data in r.iter_content(4 * MIB):
                    if offset + len(data) > size:
                        raise ValueError("Download exceeded the published file size")
                    out.write(data)
                    offset += len(data)
                    bar.update(len(data))
    if part.stat().st_size != size:
        raise ValueError(f"Incomplete download (rerun to resume): {part}")
    if sha256(part) != digest:
        # Do not leave a full-size bad partial that would fail forever on every retry.
        part.rename(part.with_name(part.name + f".bad-{time.time_ns()}"))
        raise ValueError(
            "SHA256 mismatch; corrupt partial preserved as .bad-*; rerun to download again"
        )
    part.replace(dest)
    return dest


def download_model(entry):
    paths = []
    for file in entry["files"]:
        safe_path(file["path"])
        path = model_directory(entry) / Path(file["path"]).name
        paths.append(
            download_file(
                resolve_url(entry["repo"], entry["revision"], file["path"]),
                path,
                file["size"],
                file["sha256"],
            )
        )
    # Re-read actual downloaded headers as an independent check before handing them to the packer.
    check = inspect_headers(
        [GGUFFile(p) for p in paths], [p.stat().st_size for p in paths]
    )
    if not check["compatible"]:
        raise ValueError("Downloaded model cannot run: " + "; ".join(check["reasons"]))
    return paths[0]


def local_entry(path):
    paths = unleashed.split_paths(Path(path).expanduser().resolve())
    inspection = inspect_headers(
        [GGUFFile(p) for p in paths], [p.stat().st_size for p in paths]
    )
    return {
        "id": "local/" + paths[0].stem,
        "title": "Existing local model",
        "quant": paths[0].stem,
        "provider": "local",
        "download_bytes": 0,
        "files": [],
        "inspection": inspection,
        "path": str(paths[0]),
    }


def ask_int(message, default, low, high):
    while True:
        value = input(f"{message} [{default}]: ").strip()
        try:
            n = int(value) if value else default
            if low <= n <= high:
                return n
        except ValueError:
            pass
        print(f"Enter a whole number from {low} to {high}.")


def port_available(port):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def show_scan(s):
    print(f"\nStrata Unleashed {version()} — system scan")
    print(
        f'{s["platform"]} | {s["cpu"].get("model",s["cpu"].get("processor","CPU"))} | {s["cpu"]["logical_cpus"]} threads'
    )
    print(
        f'RAM: {s["ram_total"]/GIB:.1f} GiB installed, {s["ram_available"]/GIB:.1f} GiB available'
    )
    print(f'Disk: {s["disk_free"]/GIB:.1f} GiB free for {s["storage_path"]}')
    for g in s["gpus"]:
        print(
            f'GPU {g["index"]}: {g["name"]}, {g["total_mib"]/1024:.1f} GiB VRAM, {g["free_mib"]/1024:.1f} GiB free; driver {g["driver"]}'
        )
    for err in s["errors"]:
        print(err)
    print(
        "Build tools: "
        + ", ".join(
            f'{n}: {"found" if p else "missing"}' for n, p in s["tools"].items()
        )
    )


def rows_for(entries, system, a):
    return [
        {
            "model": e,
            "assessment": assess(
                e,
                system,
                a.gpu,
                a.reserve_vram_mib,
                a.context,
                download_bytes=0 if e["provider"] == "local" else None,
            ),
        }
        for e in entries
    ]


def show_models(rows, intent):
    recs = recommendations(rows, intent)
    best = recs[0]["model"]["id"] if recs else None
    print(
        "\n #  Provider / quant                              Download   RAM est.  Status"
    )
    for n, row in enumerate(rows, 1):
        e, fit = row["model"], row["assessment"]
        label = e["id"]
        status = "RECOMMENDED" if label == best else fit["status"].upper()
        print(
            f'{n:2}  {label:46} {e["download_bytes"]/GIB:6.1f} GiB {fit["ram_estimate_bytes"]/GIB:6.1f} GiB  {status}'
        )
        for reason in fit["reasons"]:
            print(f"      {reason}")
    if intent == "uncensored" and not recs:
        print(
            "No HuiHui option fits right now. Other providers are not substituted as uncensored."
        )
    print(
        "\nISTA is preferred for compact models/coding; Atomic Chat for larger models; HuiHui for uncensored."
    )
    print(
        "Sizes are not quality scores. Header compatibility does not mean every model has been inference-tested."
    )
    return next((n for n, r in enumerate(rows, 1) if r["model"]["id"] == best), 0)


def launch_command(path, a):
    cmd = [
        sys.executable,
        "-B",
        str(ROOT / "unleashed.py"),
        "--model",
        str(path),
        "--context",
        str(a.context),
        "--port",
        str(a.port),
        "--gpu",
        str(a.gpu),
        "--reserve-vram-mib",
        str(a.reserve_vram_mib),
    ]
    for flag, enabled in (
        ("--retune", a.retune),
        ("--tune-only", a.tune_only),
        ("--build", a.build),
    ):
        if enabled:
            cmd.append(flag)
    if a.mtp:
        cmd += ["--mtp", str(Path(a.mtp).expanduser().resolve())]
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", action="version", version=version())
    ap.add_argument(
        "--scan", action="store_true", help="Show hardware only, with no model download"
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List all models with fit estimates; no downloads",
    )
    ap.add_argument(
        "--json", action="store_true", help="Emit JSON with --scan or --list"
    )
    ap.add_argument(
        "--refresh-catalog",
        action="store_true",
        help="Refresh pinned model sizes, hashes and all GGUF headers",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Use the bundled catalog; no network/downloads",
    )
    ap.add_argument(
        "--intent", choices=["small", "coder", "large", "uncensored"], default=None
    )
    ap.add_argument("--model-id", help="Exact catalog ID shown by --list")
    ap.add_argument(
        "--local-model",
        type=Path,
        help="Inspect and tune existing shards without copying or downloading them",
    )
    ap.add_argument(
        "--gpu", type=int, default=0, help="Physical NVIDIA GPU index (one GPU)"
    )
    ap.add_argument("--reserve-vram-mib", type=int, default=0)
    ap.add_argument(
        "--context", type=int, default=16384, help="Context tokens, 9216..262144"
    )
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument(
        "--mtp", type=Path, help="Optional already prepared MTP runtime folder"
    )
    ap.add_argument("--build", action="store_true", help="Rebuild engine for this GPU")
    ap.add_argument("--retune", action="store_true")
    ap.add_argument("--tune-only", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument(
        "--yes",
        action="store_true",
        help="Run the explicit --model-id/--local-model without interactive prompts",
    )
    a = ap.parse_args(argv)
    if a.mtp is not None and not a.mtp.expanduser().is_dir():
        ap.error("--mtp must name an existing prepared runtime directory")
    if (
        a.reserve_vram_mib < 0
        or a.gpu < 0
        or not 9216 <= a.context <= 262144
        or not 1 <= a.port <= 65535
    ):
        ap.error(
            "Require nonnegative reserve/GPU, context 9216..262144 and port 1..65535"
        )
    if a.json and not (a.scan or a.list):
        ap.error("--json requires --scan or --list")
    if a.offline and a.refresh_catalog:
        ap.error("--offline conflicts with --refresh-catalog")
    if a.model_id and a.local_model:
        ap.error("Choose --model-id or --local-model")
    if a.yes and not (a.model_id or a.local_model or a.scan or a.list):
        ap.error("--yes requires an explicit model")
    unleashed.configure_environment()
    system = scan_system()
    if a.scan:
        print(json.dumps(system, indent=2)) if a.json else show_scan(system)
        return 0
    if not a.json:
        show_scan(system)
    if a.refresh_catalog:
        catalog = refresh_catalog(
            ROOT / "work/model-catalog.json",
            progress=lambda s: print(s, file=sys.stderr, flush=True),
        )
    else:
        cached = ROOT / "work/model-catalog.json"
        catalog = json.loads(
            (CATALOG if a.offline or not cached.exists() else cached).read_text(
                encoding="utf-8"
            )
        )
    entries = catalog["models"]
    if a.local_model:
        entries = [local_entry(a.local_model)]
    interactive = not (a.list or a.yes)
    if interactive:
        if a.intent is None:
            print(
                "\nGoal: 1) Compact/general  2) Coding  3) Larger model  4) Uncensored"
            )
            a.intent = ["small", "coder", "large", "uncensored"][
                ask_int("Goal", 1, 1, 4) - 1
            ]
        if len(system["gpus"]) > 1:
            choices = [g["index"] for g in system["gpus"]]
            while True:
                a.gpu = ask_int("GPU index", a.gpu, min(choices), max(choices))
                if a.gpu in choices:
                    break
        a.reserve_vram_mib = ask_int(
            "Extra VRAM to reserve for other apps (MiB)", a.reserve_vram_mib, 0, 1048576
        )
        a.context = ask_int("Context tokens", a.context, 9216, 262144)
    a.intent = a.intent or "small"
    rows = rows_for(entries, system, a)
    if a.json:
        print(
            json.dumps(
                {
                    "system": system,
                    "catalog_checked_utc": catalog["checked_utc"],
                    "models": rows,
                    "recommended_ids": [
                        r["model"]["id"] for r in recommendations(rows, a.intent)
                    ],
                },
                indent=2,
            )
        )
        return 0
    print(
        f'\nCatalog checked: {catalog["checked_utc"]}. Use --refresh-catalog to check for newly published variants.'
    )
    default = show_models(rows, a.intent)
    if a.list:
        return 0
    if a.model_id:
        selected = next((r for r in rows if r["model"]["id"] == a.model_id), None)
        if not selected:
            ap.error("Unknown model ID; run --list")
    elif a.local_model:
        selected = rows[0]
    else:
        n = ask_int("Choose model (0 exits)", default, 0, len(rows))
        if not n:
            return 0
        selected = rows[n - 1]
    e, fit = selected["model"], selected["assessment"]
    if fit["reasons"]:
        print("\nCannot run this selection now:\n  " + "\n  ".join(fit["reasons"]))
        print(
            "Choose a smaller compatible model, close other apps, reduce context/reservation, or add RAM/disk as indicated."
        )
        return 2
    a.build = a.build or fit["needs_build"]
    if a.build and any(not p for p in system["tools"].values()) and not a.download_only:
        print("Install the missing source-build tools before tuning with --build.")
        return 2
    for warning in fit["warnings"]:
        print(warning)
    print(
        f'\nSelected: {e["id"]}; reserve {a.reserve_vram_mib} MiB; context {a.context}; GPU {a.gpu}'
    )
    if e["provider"] != "local":
        print(
            f'Model card and license: https://huggingface.co/{e["repo"]}/tree/{e["revision"]}'
        )
    print(
        "Procedure: verify/download shards, prepare the model, benchmark decode, then prefill, and save the best measured configuration."
    )
    if interactive:
        print(
            "Action: 1) Download, tune and serve  2) Download and tune only  3) Download only  0) Exit"
        )
        action = ask_int("Action", 1, 0, 3)
        if action == 0:
            return 0
        a.tune_only = action == 2
        a.download_only = action == 3
    elif not a.yes:
        ap.error("Noninteractive execution requires --yes and an explicit model")
    if not a.tune_only and not a.download_only:
        if interactive:
            suggested = next(
                (
                    p
                    for p in range(a.port, min(65536, a.port + 100))
                    if port_available(p)
                ),
                a.port,
            )
            a.port = ask_int("Local API port", suggested, 1, 65535)
        if not port_available(a.port):
            raise ValueError(
                f"Local port {a.port} is occupied; choose another with --port"
            )
    if a.offline and e["provider"] != "local":
        raise ValueError(
            "--offline permits scan/list or --local-model; downloads require network access"
        )
    if e["provider"] == "local":
        path = Path(e["path"])
    else:
        with download_lock():
            path = download_model(e)
    receipt = {
        "model": e["id"],
        "path": str(path),
        "reserve_vram_mib": a.reserve_vram_mib,
        "context": a.context,
        "gpu": a.gpu,
        "assessment": fit,
        "launch": launch_command(path, a),
    }
    (ROOT / "work/last-selection.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if a.download_only:
        print(f"Verified model ready: {path}")
        return 0
    return subprocess.run(launch_command(path, a), cwd=ROOT).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, requests.RequestException) as exc:
        print(f"Unleashed setup: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except (KeyboardInterrupt, EOFError):
        print(
            "\nStopped. Partial downloads can be resumed by choosing the same model again."
        )
        raise SystemExit(130)
