"""Create source/build archives without models, caches or local configuration.
Linux may include the tested build/strata. Windows is source-only unless --windows-exe
points to an actual Windows build produced and validated on Windows.
"""

import argparse, hashlib, json, os, platform, subprocess, tarfile, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.0"
DIRECTORIES = (
    "src",
    "include",
    "serve",
    "tools",
    "cmake",
    "data",
    "docs",
    "tests",
    "ref",
    "third_party",
)
FILES = (
    "CMakeLists.txt",
    "LICENSE",
    "AGENTS.md",
    "README.md",
    "UNLEASHED.md",
    "requirements-unleashed.txt",
    "requirements.txt",
    "unleashed.py",
    "start-unleashed.sh",
    "start-unleashed.ps1",
    "START-UNLEASHED.bat",
    "BUILD-UNLEASHED.ps1",
    "BUILD-UNLEASHED.sh",
    "RELEASE-NOTES.md",
)


def sources():
    out = []
    for name in FILES:
        p = ROOT / name
        if p.is_file():
            out.append(p)
    for name in DIRECTORIES:
        for p in (ROOT / name).rglob("*"):
            if not p.is_file() or p.is_symlink():
                continue
            if any(
                part in ("__pycache__", ".git", ".pytest_cache", "node_modules")
                or part.startswith("build-")
                for part in p.relative_to(ROOT).parts
            ):
                continue
            if p.suffix in (
                ".pyc",
                ".log",
                ".rej",
                ".o",
                ".obj",
                ".exe",
                ".dll",
                ".so",
            ):
                continue
            out.append(p)
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows-exe", type=Path)
    a = ap.parse_args()
    out = ROOT / "releases"
    out.mkdir(exist_ok=True)
    base = sources()
    srcid = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    hashes = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in base
    }
    manifests = {}
    for target in ("linux-x86_64", "windows-x86_64"):
        files = [(p, p.relative_to(ROOT).as_posix()) for p in base]
        binary = ROOT / "build/strata" if target.startswith("linux") else a.windows_exe
        if binary and binary.is_file():
            files.append(
                (
                    binary,
                    "build/"
                    + ("strata.exe" if target.startswith("windows") else "strata"),
                )
            )
        manifest = {
            "name": "Strata Unleashed",
            "version": VERSION,
            "target": target,
            "upstream_commit": srcid,
            "kind": (
                "binary-and-source" if binary and binary.is_file() else "source-build"
            ),
            "source_sha256": hashes,
            "binary_sha256": (
                hashlib.sha256(binary.read_bytes()).hexdigest()
                if binary and binary.is_file()
                else None
            ),
            "validation": (
                "Linux Q5 and CUDA numerical checks; see RELEASE-NOTES.md"
                if target.startswith("linux")
                else "Windows build/run not validated in the Linux development environment"
            ),
            "requirements": {
                "python": "3.10+",
                "gpu": "NVIDIA CUDA; bundled Linux engine targets architecture 89",
                "linux_binary_abi": (
                    "glibc 2.43; GLIBCXX_3.4.32; AVX2/FMA/F16C/BMI2 CPU; rebuild on older systems"
                    if target.startswith("linux")
                    else None
                ),
                "cuda_toolkit": "13.1 used in Linux validation; needed to build or supply runtime libraries",
            },
            "build_host": platform.platform() if target.startswith("linux") else None,
        }
        manifest_path = out / (target + "-manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        files.append((manifest_path, "MANIFEST.json"))
        top = f"StrataUnleashed-{VERSION}"
        if target.startswith("windows"):
            suffix = "" if binary and binary.is_file() else "-source"
            dest = out / f"{top}-{target}{suffix}.zip"
            with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                for p, name in files:
                    info = zipfile.ZipInfo(top + "/" + name, (2026, 10, 3, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = (0o755 if p.suffix == ".sh" else 0o644) << 16
                    z.writestr(info, p.read_bytes())
        else:
            dest = out / f"{top}-{target}.tar.gz"
            with tarfile.open(dest, "w:gz", compresslevel=6) as tar:
                for p, name in files:
                    info = tar.gettarinfo(str(p), top + "/" + name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 1790985600
                    with p.open("rb") as f:
                        tar.addfile(info, f)
        manifests[dest.name] = {
            "bytes": dest.stat().st_size,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        }
    (out / "SHA256SUMS").write_text(
        "".join(f"{v['sha256']}  {name}\n" for name, v in manifests.items())
    )
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
