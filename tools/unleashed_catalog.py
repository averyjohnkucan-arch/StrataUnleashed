"""Pinned Hugging Face catalogs and GGUF-header compatibility checks (no weight downloads)."""

from __future__ import annotations

import concurrent.futures
import datetime
import io
import json
import re
import struct
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests
from tools.gguf_reader import GGUFFile, BLOCK_GEOMETRY

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/unleashed-models.json"
SOURCES = {
    "ista": ("ISTA Flash Next", "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
    "ista-coder": (
        "ISTA Flash Next Coder",
        "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF",
    ),
    "atomic": ("Atomic Chat", "AtomicChat/Qwen3.8-Flash-Next-GGUF"),
    "huihui": (
        "HuiHui uncensored / abliterated",
        "huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF",
    ),
    "unsloth": ("Unsloth", "unsloth/Qwen3.8-Flash-Next-GGUF"),
}
SPLIT = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")
EXPERT = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
EXPERT_TYPES = set(
    "Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q4_K Q5_K Q6_K IQ2_XXS IQ2_XS IQ3_XXS IQ3_S IQ2_S IQ4_NL IQ4_XS IQ1_M Q2_0 MXFP4 NVFP4".split()
)
DENSE_TYPES = EXPERT_TYPES | {"Q3_K", "F32", "F16", "BF16"}
PLE_TYPES = set("F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 IQ4_NL".split())
GUARD = {
    "block_count": 48,
    "embedding_length": 2560,
    "attention.head_count": 24,
    "attention.head_count_kv": 2,
}


def safe_path(name):
    p = PurePosixPath(name)
    if (
        not name
        or p.is_absolute()
        or any(x in ("..", ".", "") for x in name.split("/"))
        or "\\" in name
        or ":" in name
    ):
        raise ValueError(f"Unsafe catalog path: {name}")
    return p


def resolve_url(repo, revision, name):
    safe_path(name)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or not re.fullmatch(
        r"[0-9a-f]{40}", revision
    ):
        raise ValueError(
            "Model download requires a repository and immutable commit revision"
        )
    return f'https://huggingface.co/{repo}/resolve/{revision}/{quote(name, safe="/")}'


def variants(provider, info):
    """Group all complete GGUF shard sets, excluding projector/MTP/imatrix artifacts."""
    groups = {}
    for item in info["siblings"]:
        name = item["rfilename"]
        if not name.endswith(".gguf") or any(
            x in name.lower() for x in ("mmproj", "imatrix", "mtp")
        ):
            continue
        safe_path(name)
        match = SPLIT.match(name)
        key = match[1] if match else name[:-5]
        groups.setdefault(key, []).append(item)
    result = []
    for key, files in sorted(groups.items()):
        files.sort(key=lambda f: f["rfilename"])
        first = SPLIT.match(files[0]["rfilename"])
        if first:
            count = int(first[3])
            if len(files) != count or [
                int(SPLIT.match(f["rfilename"])[2]) for f in files
            ] != list(range(1, count + 1)):
                raise ValueError(f"Incomplete remote shard set: {key}")
        quant = PurePosixPath(key).parent.name or PurePosixPath(key).name
        if provider == "atomic":
            quant = re.sub(r"^Qwen3\.8-Flash-Next-", "", quant)
        parts = []
        for f in files:
            digest = f.get("lfs", {}).get("sha256")
            size = f.get("size")
            if (
                not isinstance(size, int)
                or size <= 0
                or not digest
                or not re.fullmatch("[0-9a-f]{64}", digest)
            ):
                raise ValueError(f'Missing published size/SHA256: {f["rfilename"]}')
            parts.append({"path": f["rfilename"], "size": size, "sha256": digest})
        result.append(
            {
                "id": f"{provider}/{quant}",
                "provider": provider,
                "title": SOURCES[provider][0],
                "quant": quant,
                "repo": SOURCES[provider][1],
                "revision": info["sha"],
                "files": parts,
                "download_bytes": sum(f["size"] for f in parts),
            }
        )
    return result


def remote_header(entry, file, cache):
    """Bounded prefix reads; never download a multi-GB shard to inspect its directory."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (file["sha256"] + ".header")
    for limit in (65536, 1048576, 16777216, 33554432):
        if path.exists() and path.stat().st_size >= min(limit, file["size"]):
            raw = path.read_bytes()
        else:
            url = resolve_url(entry["repo"], entry["revision"], file["path"])
            with requests.get(
                url,
                headers={"Range": f"bytes=0-{limit-1}", "Accept-Encoding": "identity"},
                stream=True,
                timeout=(15, 60),
            ) as r:
                r.raise_for_status()
                if r.status_code == 206 and not r.headers.get(
                    "Content-Range", ""
                ).startswith("bytes 0-"):
                    raise ValueError("Header request returned the wrong byte range")
                raw = r.raw.read(limit)
            path.write_bytes(raw)
        g = GGUFFile.__new__(GGUFFile)
        g.path, g.metadata, g.tensors, g.alignment = Path(file["path"]), {}, [], 32
        try:
            g._parse(io.BytesIO(raw))
            g.data_start = g._data_start
            return g
        except (struct.error, IndexError, UnicodeError):
            continue
    raise ValueError(f'GGUF header exceeds 32 MiB or is truncated: {file["path"]}')


def inspect_headers(headers, sizes):
    """Conservative engine contract check and weight accounting across every shard."""
    from tools.iq_pack import check_split, form_of, FLOAT, NATIVE_PLE_KEY

    check_split(headers)
    md = headers[0].metadata
    reasons = []
    if md.get("general.architecture") != "qwen4exp":
        reasons.append("Architecture is not qwen4exp")
    for suffix, expected in GUARD.items():
        if md.get("qwen4exp." + suffix) != expected:
            reasons.append(f"{suffix} must be {expected}")
    n_expert = md.get("qwen4exp.expert_count", 0)
    if not isinstance(n_expert, int) or not 0 < n_expert <= 512:
        reasons.append("Expert count must be 1..512")
    used = md.get("qwen4exp.expert_used_count", 0)
    if not isinstance(used, int) or not 0 < used <= n_expert:
        reasons.append("Invalid experts used per token")
    tensors, expert_types, ple_types = {}, set(), set()
    expert_bytes = ple_bytes = dense_bytes = pack_bytes = 0
    host_embedding_bytes = gpu_dense_bytes = raw_dense_bytes = 0
    for g, size in zip(headers, sizes):
        for t in g.tensors:
            if t.name in tensors:
                reasons.append(f"Duplicate tensor: {t.name}")
            tensors[t.name] = t
            nb = t.expected_bytes()
            if nb is None or g.data_start + t.offset + (nb or 0) > size:
                reasons.append(
                    f"Unknown format, invalid shape or truncated tensor: {t.name}"
                )
                continue
            if EXPERT.match(t.name):
                expert_bytes += nb
                expert_types.add(t.type_name)
                if t.type_name not in EXPERT_TYPES:
                    reasons.append(f"Unsupported expert encoding: {t.type_name}")
                block = BLOCK_GEOMETRY[t.type_name][0]
                if len(t.shape) != 3 or t.shape[-1] != n_expert or t.shape[0] % block:
                    reasons.append(f"Invalid expert geometry: {t.name}")
            elif t.name == "per_layer_token_embd.weight":
                ple_bytes += nb
                ple_types.add(t.type_name)
                row = nb // t.shape[1] if len(t.shape) == 2 and t.shape[1] else 0
                if t.type_name not in PLE_TYPES or not 0 < row <= 640:
                    reasons.append(
                        f"Unsupported PLE table: {t.type_name}, row {row} bytes"
                    )
            else:
                form = form_of(t.name)
                native = t.type_name not in FLOAT and (
                    form is None
                    or (
                        t.name == "blk.1.ple_key.weight"
                        and t.type_name in NATIVE_PLE_KEY
                    )
                )
                if native and t.type_name not in DENSE_TYPES:
                    reasons.append(f"Unsupported dense encoding: {t.type_name}")
                stored = (
                    nb
                    if native
                    else t.elements * (4 if (form or t.type_name) == "F32" else 2)
                )
                dense_bytes += stored
                raw_dense_bytes += nb
                if t.name == "token_embd.weight":
                    host_embedding_bytes += nb
                else:
                    gpu_dense_bytes += stored
                if not native:
                    pack_bytes += stored + 64
    for layer in range(48):
        ts = [
            tensors.get(f"blk.{layer}.ffn_{role}_exps.weight")
            for role in ("gate", "up", "down")
        ]
        if any(t is None for t in ts):
            reasons.append(f"Missing expert roles in layer {layer}")
        elif ts[0].type_name != ts[1].type_name or ts[0].shape != ts[1].shape:
            reasons.append(f"Gate/up mismatch in layer {layer}")
        elif ts[0].shape[0] != 2560 or ts[2].shape[:2] != [ts[0].shape[1], 2560]:
            reasons.append(f"Invalid gate/down dimensions in layer {layer}")
    if not ple_bytes:
        reasons.append("Missing PLE table")
    return {
        "compatible": not reasons,
        "reasons": sorted(set(reasons)),
        "memory_accounting_version": 2,
        "host_embedding_bytes": host_embedding_bytes,
        "gpu_dense_bytes": gpu_dense_bytes,
        "raw_dense_bytes": raw_dense_bytes,
        "expert_bytes": expert_bytes,
        "ple_bytes": ple_bytes,
        "dense_bytes": dense_bytes,
        "pack_bytes": pack_bytes,
        "expert_types": sorted(expert_types),
        "ple_types": sorted(ple_types),
        "experts": n_expert,
        "validation": "All shard headers checked; runtime quality and speed are not implied",
    }


def refresh_catalog(
    destination=CATALOG, cache=ROOT / "work/catalog-headers", progress=print
):
    entries = []
    for provider, (_, repo) in SOURCES.items():
        progress(f"Reading {repo} ...")
        r = requests.get(
            "https://huggingface.co/api/models/" + repo,
            params={"blobs": "true"},
            timeout=40,
        )
        r.raise_for_status()
        for entry in variants(provider, r.json()):
            progress(f'  Inspecting {entry["id"]}: {len(entry["files"])} shard headers')
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                headers = list(
                    pool.map(lambda f: remote_header(entry, f, cache), entry["files"])
                )
            entry["inspection"] = inspect_headers(
                headers, [f["size"] for f in entry["files"]]
            )
            entries.append(entry)
    result = {
        "schema": 1,
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "models": entries,
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temp.replace(destination)
    return result


if __name__ == "__main__":
    refresh_catalog()
