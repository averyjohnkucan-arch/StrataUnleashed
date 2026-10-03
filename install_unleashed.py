"""Install the latest verified release into one folder, then open model setup.
Uses only the Python standard library. Existing installations are launched, not overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

REPOSITORY = "averyjohnkucan-arch/StrataUnleashed"
API = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"


def fetch(url, destination):
    request = urllib.request.Request(
        url, headers={"User-Agent": "StrataUnleashed-installer"}
    )
    with urllib.request.urlopen(request, timeout=120) as response, Path(
        destination
    ).open("wb") as output:
        shutil.copyfileobj(response, output)


def safe_members(names):
    for name in names:
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
            raise ValueError(f"Unsafe archive member: {name}")


def extract(archive, destination):
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            safe_members(z.namelist())
            if any(
                (i.external_attr >> 16) & 0o170000 == 0o120000 for i in z.infolist()
            ):
                raise ValueError("Archive contains a symbolic link")
            z.extractall(destination)
    else:
        with tarfile.open(archive) as t:
            safe_members(m.name for m in t.getmembers())
            if any(not (m.isfile() or m.isdir()) for m in t.getmembers()):
                raise ValueError("Archive contains a special file or link")
            # Manual regular-file extraction works on Python 3.10 as well as newer releases.
            for m in t.getmembers():
                p = destination / m.name
                if m.isdir():
                    p.mkdir(parents=True, exist_ok=True)
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with t.extractfile(m) as source, p.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    p.chmod(m.mode & 0o777)


def install(destination, release=None):
    destination = Path(destination).expanduser().resolve()
    if (destination / "cli-unleashed.sh").is_file() and (
        destination / "unleashed.py"
    ).is_file():
        print(f"Using your existing installation: {destination}")
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    if any(p.name != "work" for p in destination.iterdir()):
        raise ValueError(
            f"{destination} contains other files. Choose an empty folder; nothing was overwritten."
        )
    work = destination / "work/installer"
    work.mkdir(parents=True, exist_ok=True)
    if release is None:
        fetch(API, work / "release.json")
        release = json.loads((work / "release.json").read_text())
    ending = "-windows-x86_64-source.zip" if os.name == "nt" else "-linux-x86_64.tar.gz"
    asset = next(
        (
            a
            for a in release["assets"]
            if a["name"].startswith("StrataUnleashed-") and a["name"].endswith(ending)
        ),
        None,
    )
    sums = next((a for a in release["assets"] if a["name"] == "SHA256SUMS"), None)
    if asset is None or sums is None:
        raise ValueError(
            "The latest release is missing an installer archive or checksums"
        )
    safe_members([asset["name"]])
    archive = work / asset["name"]
    print(f'Downloading Strata Unleashed {release["tag_name"]} ...', flush=True)
    fetch(asset["browser_download_url"], archive)
    fetch(sums["browser_download_url"], work / "SHA256SUMS")
    checks = dict(
        (line.split()[1], line.split()[0])
        for line in (work / "SHA256SUMS").read_text().splitlines()
    )
    if hashlib.sha256(archive.read_bytes()).hexdigest() != checks.get(asset["name"]):
        raise ValueError("Release checksum failed; archive was not installed")
    stage = work / "unpacked"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    extract(archive, stage)
    folders = list(stage.iterdir())
    if len(folders) != 1 or not (folders[0] / "unleashed.py").is_file():
        raise ValueError("Unexpected release layout")
    for source in folders[0].iterdir():
        shutil.move(str(source), destination / source.name)
    print(
        f"Installed in {destination}. Next: choose and download your model.", flush=True
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=Path.home() / "Documents/StrataUnleashed"
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Install only; do not start model setup",
    )
    args, forwarded = parser.parse_known_args()
    root = install(args.directory)
    if args.no_launch:
        return 0
    command = (
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(root / "start-unleashed.ps1"),
            "--wizard",
        ]
        if os.name == "nt"
        else ["bash", str(root / "cli-unleashed.sh")]
    )
    return subprocess.run([*command, *forwarded], cwd=root).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        print(f"Setup stopped: {exc}", file=sys.stderr)
        raise SystemExit(1)
