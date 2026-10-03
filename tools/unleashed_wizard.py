"""Interactive model selection, hardware advice, verified downloads and procedural tuning."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path

import requests
from tools.unleashed_download import download_lock, download_model
from tools.unleashed_hardware import scan_system, assess, recommendations

import unleashed
from tools.unleashed_catalog import (
    CATALOG,
    ROOT,
    inspect_headers,
    refresh_catalog,
)
from tools.gguf_reader import GGUFFile
from tools.unleashed_storage import scan_storage

from tools.unleashed_policy import NATIVE_CONTEXT, KV_FORMATS, default_kv

GIB = 1024**3


def version():
    return (ROOT / "UNLEASHED_VERSION").read_text().strip()


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


def show_storage(storage):
    print(f"Drive: {storage.get('device', storage['path'])} ({storage['transport']})")
    if storage.get("read_mib_s") is not None:
        print(
            f"Read-speed sample: {storage['read_mib_s']:.0f} MiB/s (OS cache bypassed)."
        )
    else:
        print(storage.get("speed_note", "Read speed is not available."))
    for warning in storage.get("warnings", []):
        print(f"Note: {warning}")


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
        print(
            f'  Default KV: {default_kv(g["total_mib"])}; native context: {NATIVE_CONTEXT} tokens'
        )
    if s.get("storage"):
        show_storage(s["storage"])
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
                kv=getattr(a, "kv", None),
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
        status = (
            "Recommended"
            if label == best
            else ("Fits estimate" if fit["status"] == "candidate" else "Not available")
        )
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
        ("--chat", a.chat),
    ):
        if enabled:
            cmd.append(flag)
    if getattr(a, "kv", None):
        cmd += ["--kv", a.kv]
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
        "--context",
        type=int,
        default=NATIVE_CONTEXT,
        choices=[NATIVE_CONTEXT],
        help="Native context: 262144 tokens",
    )
    ap.add_argument(
        "--kv", choices=KV_FORMATS, help="Override the GPU-capacity KV default"
    )
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument(
        "--mtp", type=Path, help="Optional already prepared MTP runtime folder"
    )
    ap.add_argument("--build", action="store_true", help="Rebuild engine for this GPU")
    ap.add_argument(
        "--chat",
        action="store_true",
        help="Open terminal test chat with no system prompt",
    )
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
        or a.context != NATIVE_CONTEXT
        or not 1 <= a.port <= 65535
    ):
        ap.error("Require nonnegative reserve/GPU, context 262144 and port 1..65535")
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
        system["storage"] = scan_storage(a.local_model.parent, a.local_model)
        if not a.json:
            show_storage(system["storage"])
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
        print("Native context: 262144 tokens for tuning and inference.")
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
            "Choose a smaller compatible model, close other apps, reduce reservation, or add RAM/disk as indicated."
        )
        return 2
    a.build = a.build or fit["needs_build"]
    if a.build and any(not p for p in system["tools"].values()) and not a.download_only:
        print("Install the missing source-build tools before tuning with --build.")
        return 2
    for warning in fit["warnings"]:
        print(warning)
    print(
        f'\nSelected: {e["id"]}; reserve {a.reserve_vram_mib} MiB; context {a.context}; KV {fit.get("kv", a.kv or "auto")}; GPU {a.gpu}'
    )
    if e["provider"] != "local":
        print(
            f'Model card and license: https://huggingface.co/{e["repo"]}/tree/{e["revision"]}'
        )
    print(
        "We will download the model, check its files, and find settings that work well on your PC."
    )
    if interactive:
        print(
            "Next: 1) Download, set up and chat  2) Download, set up and open the API  3) Set up only  4) Download only  0) Exit"
        )
        action = ask_int("Next", 1, 0, 4)
        if action == 0:
            return 0
        a.chat = action == 1
        a.tune_only = action == 3
        a.download_only = action == 4
    elif not a.yes:
        ap.error("Noninteractive execution requires --yes and an explicit model")
    if not a.tune_only and not a.download_only and not a.chat:
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
    model_files = unleashed.split_paths(path)
    sample = max(model_files, key=lambda p: p.stat().st_size)
    storage = scan_storage(path.parent, sample)
    show_storage(storage)
    (ROOT / "work/storage-check.json").write_text(json.dumps(storage, indent=2) + "\n")
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
    print(
        "Preparing and tuning your model. The first setup can take a while; later starts reuse the result.",
        flush=True,
    )
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
