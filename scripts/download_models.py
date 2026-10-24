"""Download the converted RKNN models (Rockchip NPU) from the GitHub release and verify their checksums.

    python scripts/download_models.py              # the model for the board this runs on
    python scripts/download_models.py rk3576       # a specific SoC: rk3566 | rk3568 | rk3576 | rk3588
    python scripts/download_models.py all          # all three models (~720 MB)
    python scripts/download_models.py --int8 ...   # the experimental int8 models instead of fp16

The models are too large for git (fp16 221-253 MB, int8 ~150 MB each), so they are published as assets of the release
MODELS_TAG (default rknn-models-v1) and downloaded over plain HTTPS. No GitHub account is needed. Set GITHUB_TOKEN
only to download from a private fork. Only the standard library is used, so this runs before anything else is
installed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

REPO = os.environ.get("MODELS_REPO", "anand34577/saywho")
TAG = os.environ.get("MODELS_TAG", "rknn-models-v1")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.environ.get("RKNN_DIR", os.path.join(HERE, "rknn_models"))
SOCS = ("rk3566", "rk3576", "rk3588")
ALIASES = {"rk3568": "rk3566"}  # RK3568 runs the RK3566 model


def detect_soc() -> str | None:
    forced = os.environ.get("RK_PLATFORM")
    if forced:
        return forced.lower()
    for path in ("/proc/device-tree/compatible", "/sys/firmware/devicetree/base/compatible"):
        try:
            with open(path, "rb") as f:
                compatible = f.read().decode(errors="ignore").lower()
        except OSError:
            continue
        for name in ("rk3588", "rk3576", "rk3568", "rk3566"):
            if name in compatible:
                return name
    return None


def checksums() -> dict:
    with open(os.path.join(DEST, "SHA256SUMS")) as f:
        return {name: digest for digest, name in (line.split() for line in f if line.strip())}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def via_https(name: str) -> bool:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = {"User-Agent": "saywho"}
    if token:
        # private forks: resolve the asset through the API, then download it as a binary stream
        headers["Authorization"] = f"Bearer {token}"
        api = f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}"
        with urllib.request.urlopen(urllib.request.Request(api, headers=headers)) as r:
            assets = {a["name"]: a["url"] for a in json.load(r)["assets"]}
        url = assets[name]
        headers["Accept"] = "application/octet-stream"
    else:
        url = f"https://github.com/{REPO}/releases/download/{TAG}/{name}"
    tmp = os.path.join(DEST, name + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        for block in iter(lambda: r.read(1 << 20), b""):
            f.write(block)
            done += len(block)
            if total:
                print(f"\r  {done / 1e6:6.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    os.replace(tmp, os.path.join(DEST, name))
    return True


def fetch(name: str, expected: str) -> None:
    path = os.path.join(DEST, name)
    if os.path.exists(path) and sha256(path) == expected:
        print(f"{name}: already present, checksum OK")
        return
    print(f"{name}: downloading from {REPO} release {TAG}")
    try:
        via_https(name)
    except (urllib.error.URLError, KeyError) as e:
        sys.exit(f"{name}: download failed ({e}). Check the internet connection and run this again; files that "
                 "finished downloading are kept.")
    if sha256(path) != expected:
        os.remove(path)
        sys.exit(f"{name}: checksum mismatch, file removed. Try again.")
    print(f"{name}: OK")


def main():
    args = [a.lower() for a in sys.argv[1:]]
    if "-h" in args or "--help" in args:
        print(__doc__)
        return
    suffix = "_int8" if "--int8" in args else ""
    positional = [a for a in args if not a.startswith("--")]
    arg = positional[0] if positional else ""
    if arg == "all":
        socs = list(SOCS)
    else:
        soc = arg or detect_soc()
        if not soc:
            sys.exit("No Rockchip SoC detected. Name one: rk3566 | rk3568 | rk3576 | rk3588 | all")
        soc = ALIASES.get(soc, soc)
        if soc not in SOCS:
            sys.exit(f"Unknown SoC {soc!r}. Choose rk3566 | rk3568 | rk3576 | rk3588 | all")
        socs = [soc]
    os.makedirs(DEST, exist_ok=True)
    sums = checksums()
    for soc in socs:
        name = f"nemotron3_diarization_{soc}{suffix}.rknn"
        if name not in sums:
            sys.exit(f"{name}: not published in release {TAG} (no checksum in rknn_models/SHA256SUMS)")
        fetch(name, sums[name])


if __name__ == "__main__":
    main()
