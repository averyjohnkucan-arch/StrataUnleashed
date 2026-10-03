"""Read-only drive identification and bounded, uncached read-speed checks."""

from __future__ import annotations

import ctypes
import json
import mmap
import os
import subprocess
import time
from pathlib import Path

MIB = 1024**2


def linux_drive(path):
    result = subprocess.run(
        [
            "findmnt",
            "--json",
            "--target",
            str(path),
            "--output",
            "SOURCE,FSTYPE,TARGET",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    mount = json.loads(result.stdout)["filesystems"][0]
    source = mount["source"].split("[")[0]
    result = subprocess.run(
        ["lsblk", "--json", "--inverse", "--output", "NAME,TYPE,TRAN,ROTA", source],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    leaves = []

    def visit(node):
        children = node.get("children", [])
        if children:
            for child in children:
                visit(child)
        else:
            leaves.append(node)

    for node in json.loads(result.stdout)["blockdevices"]:
        visit(node)
    transports = {str(n.get("tran") or "unknown").lower() for n in leaves}
    nvme = transports == {"nvme"}
    return {
        "device": source,
        "filesystem": mount["fstype"],
        "transport": ",".join(sorted(transports)),
        "nvme": nvme,
        "rotational": any(bool(n.get("rota")) for n in leaves),
    }


def windows_drive(path):
    # Only a validated, single-letter drive identifier enters the PowerShell command.
    drive = Path(path).resolve().drive.rstrip(":")
    if len(drive) != 1 or not drive.isalpha():
        raise ValueError("The model is not on a local Windows drive letter")
    script = f"Get-Partition -DriveLetter '{drive}' | Get-Disk | Select-Object Number,FriendlyName,BusType | ConvertTo-Json -Compress"
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    data = json.loads(result.stdout)
    if isinstance(data, list):
        data = data[0]
    bus = str(data.get("BusType", "unknown"))
    # Get-Disk may serialize its enum as the documented STORAGE_BUS_TYPE numeric value.
    bus = {
        "17": "NVMe",
        "11": "SATA",
        "7": "USB",
        "1": "SCSI",
        "8": "RAID",
        "0": "unknown",
    }.get(bus, bus)
    return {
        "device": data.get("FriendlyName", drive + ":"),
        "transport": bus,
        "nvme": bus.lower() == "nvme",
        "rotational": None,
    }


def direct_reads(path, budget=32 * MIB):
    """Measure a bounded sequential sample with the OS cache bypassed, or return unavailable."""
    size = Path(path).stat().st_size
    block = MIB
    count = min(budget // block, size // block)
    if count < 4:
        raise ValueError(
            "Need an existing file of at least 4 MiB for the read-speed check"
        )
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        kernel.CreateFileW.restype = ctypes.c_void_p
        kernel.VirtualAlloc.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        kernel.VirtualAlloc.restype = ctypes.c_void_p
        kernel.ReadFile.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.VirtualFree.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
        ]
        handle = kernel.CreateFileW(
            str(Path(path).resolve()), 0x80000000, 7, None, 3, 0x20000000, None
        )
        if handle in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_last_error(), "Cannot open unbuffered read handle")
        buffer = kernel.VirtualAlloc(None, block, 0x3000, 0x04)
        try:
            if not buffer:
                raise OSError("Cannot allocate an aligned read buffer")
            total = 0
            started = time.perf_counter()
            for _ in range(count):
                received = ctypes.c_uint32()
                if not kernel.ReadFile(
                    handle, buffer, block, ctypes.byref(received), None
                ):
                    raise OSError(ctypes.get_last_error(), "Unbuffered read failed")
                total += received.value
            elapsed = time.perf_counter() - started
        finally:
            if buffer:
                kernel.VirtualFree(buffer, 0, 0x8000)
            kernel.CloseHandle(handle)
    elif hasattr(os, "O_DIRECT") and hasattr(os, "readv"):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        try:
            with mmap.mmap(-1, block) as buffer:
                started = time.perf_counter()
                total = 0
                for _ in range(count):
                    total += os.readv(fd, [buffer])
                elapsed = time.perf_counter() - started
        finally:
            os.close(fd)
    else:
        raise ValueError(
            "Uncached read-speed measurement is unavailable on this system"
        )
    if total != count * block:
        raise ValueError("The read-speed sample was truncated")
    return {
        "read_mib_s": round(total / MIB / max(elapsed, 0.000001), 1),
        "sample_bytes": total,
        "cache_bypassed": True,
        "method": "bounded sequential read; not a random-I/O inference benchmark",
    }


def scan_storage(directory, sample=None):
    directory = Path(directory).resolve()
    while not directory.exists():
        directory = directory.parent
    result = {
        "path": str(directory),
        "transport": "unknown",
        "nvme": None,
        "read_mib_s": None,
        "warnings": [],
    }
    try:
        result.update(
            windows_drive(directory) if os.name == "nt" else linux_drive(directory)
        )
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        result["identification_error"] = str(exc)
    if result["nvme"] is not True:
        result["warnings"].append(
            "This drive is not identified as NVMe. Your experience may vary: model loading and SSD-backed inference can be slower, especially on SATA, USB or a hard disk."
        )
    if sample:
        try:
            result.update(direct_reads(sample))
        except (OSError, ValueError) as exc:
            result["speed_note"] = str(exc)
    else:
        result["speed_note"] = (
            "Read speed will be checked after a model file is available; no test files are written."
        )
    if result["read_mib_s"] is not None and result["read_mib_s"] < 500:
        result["warnings"].append(
            "Measured sequential reading is below 500 MiB/s. Loading and disk-backed inference may be slower; this threshold is advisory, not a compatibility limit."
        )
    return result
