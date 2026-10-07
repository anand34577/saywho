"""Download Rockchip's NPU runtime (librknnrt.so and rknn-toolkit-lite2 2.3.2) directly from Rockchip.

This project does not redistribute these files: they are under Rockchip's proprietary RKNN SDK License,
https://github.com/airockchip/rknn-toolkit2/blob/master/LICENSE. Running this script with --accept-license
(or RKNN_ACCEPT_LICENSE=yes) means you accept that license.

    python3 scripts/fetch_rknn_runtime.py --accept-license --target /opt/rknn      # Docker: lib/ + python/
    python3 scripts/fetch_rknn_runtime.py --accept-license --python-only           # into this Python (venv)
    sudo python3 scripts/fetch_rknn_runtime.py --accept-license --lib-only --lib-dir /usr/lib

Re-running is safe: files already present with the right checksum are kept. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

VERSION = "2.3.2"
LICENSE_URL = "https://github.com/airockchip/rknn-toolkit2/blob/master/LICENSE"
LIB_URL = f"https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v{VERSION}/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"
LIB_SHA256 = "d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8"
PYPI_JSON = f"https://pypi.org/pypi/rknn-toolkit-lite2/{VERSION}/json"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: str, expected: str) -> None:
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "nemotron-diarization"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        for block in iter(lambda: r.read(1 << 20), b""):
            f.write(block)
    if sha256(tmp) != expected:
        os.remove(tmp)
        raise SystemExit(f"checksum mismatch for {url}")
    os.replace(tmp, dest)


def fetch_lib(lib_dir: str) -> None:
    dest = os.path.join(lib_dir, "librknnrt.so")
    if os.path.exists(dest) and sha256(dest) == LIB_SHA256:
        print(f"librknnrt.so {VERSION}: present ({dest})")
        return
    os.makedirs(lib_dir, exist_ok=True)
    print(f"librknnrt.so {VERSION}: downloading from github.com/airockchip/rknn-toolkit2")
    download(LIB_URL, dest, LIB_SHA256)
    print(f"librknnrt.so {VERSION}: installed to {dest}")


def fetch_python(target: str | None) -> None:
    if target and os.path.isdir(os.path.join(target, "rknnlite")):
        print(f"rknn-toolkit-lite2 {VERSION}: present ({target})")
        return
    if not target:
        try:
            from importlib.metadata import version

            if version("rknn-toolkit-lite2") == VERSION:
                print(f"rknn-toolkit-lite2 {VERSION}: present")
                return
        except Exception:  # noqa: BLE001 - not installed
            pass
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    with urllib.request.urlopen(PYPI_JSON, timeout=60) as r:
        files = [u for u in json.load(r)["urls"] if f"-{tag}-" in u["filename"] and "aarch64" in u["filename"]]
    if not files:
        raise SystemExit(f"rknn-toolkit-lite2 {VERSION} has no wheel for Python {sys.version_info.major}."
                         f"{sys.version_info.minor} on aarch64 (supported: 3.7-3.12)")
    wheel = files[0]
    print(f"rknn-toolkit-lite2 {VERSION}: downloading {wheel['filename']} from PyPI")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, wheel["filename"])
        download(wheel["url"], path, wheel["digests"]["sha256"])
        cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--disable-pip-version-check", path]
        if target:
            cmd += ["--target", target]
        subprocess.run(cmd, check=True)
    print(f"rknn-toolkit-lite2 {VERSION}: installed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accept-license", action="store_true", help=f"accept Rockchip's RKNN SDK License ({LICENSE_URL})")
    ap.add_argument("--target", help="install into TARGET/lib and TARGET/python (Docker)")
    ap.add_argument("--lib-dir", help="where to put librknnrt.so (default: TARGET/lib)")
    ap.add_argument("--lib-only", action="store_true")
    ap.add_argument("--python-only", action="store_true")
    a = ap.parse_args()

    if not (a.accept_license or os.environ.get("RKNN_ACCEPT_LICENSE", "").lower() in ("yes", "true", "1")):
        sys.exit(f"The Rockchip NPU runtime is licensed by Rockchip under the RKNN SDK License:\n  {LICENSE_URL}\n"
                 "Read it, then re-run with --accept-license (or set RKNN_ACCEPT_LICENSE=yes) to download it.")
    lib_dir = a.lib_dir or (os.path.join(a.target, "lib") if a.target else None)
    if not a.python_only:
        if not lib_dir:
            sys.exit("give --target or --lib-dir")
        fetch_lib(lib_dir)
    if not a.lib_only:
        fetch_python(os.path.join(a.target, "python") if a.target else None)


if __name__ == "__main__":
    main()
