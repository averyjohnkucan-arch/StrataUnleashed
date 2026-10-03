"""Download pinned model shards with resume, checksum verification and a process lock."""

from __future__ import annotations
import hashlib
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
import requests
from tqdm import tqdm
from tools.unleashed_catalog import ROOT, inspect_headers, resolve_url, safe_path
from tools.gguf_reader import GGUFFile

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
