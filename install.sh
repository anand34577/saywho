#!/usr/bin/env bash
# One-step install for Linux and macOS (x86_64 / arm64, incl. Rockchip boards without Docker).
#
#   ./install.sh            auto-detect: Rockchip NPU -> NVIDIA GPU -> CPU
#   ./install.sh cpu        force a CPU-only install (smallest download)
#   ./install.sh cuda       force the NVIDIA CUDA build of PyTorch
#   ./install.sh rknn       Rockchip NPU (RK3566/RK3568/RK3576/RK3588, needs the vendor rknpu kernel driver)
#
# Env: PYTHON=python3.11 to pick the interpreter, TORCH_CUDA=cu128 to pick the CUDA wheel (default cu126).
# Re-running is safe: the existing .venv, PyTorch and .env are kept.
set -euo pipefail
cd "$(dirname "$0")"

TARGET="${1:-auto}"
PY="${PYTHON:-python3}"
TORCH_CUDA="${TORCH_CUDA:-cu126}"
say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

command -v "$PY" >/dev/null || die "$PY not found. Install Python 3.10-3.13 (e.g. sudo apt install python3 python3-venv)."
"$PY" -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] <= (3, 13)))' \
  || die "Python 3.10-3.13 required, found $("$PY" --version 2>&1). Set PYTHON=python3.11 (or similar)."

if [ "$TARGET" = auto ]; then
  if tr -d '\0' < /proc/device-tree/compatible 2>/dev/null | grep -qE 'rk35(66|68|76|88)' && [ "$(uname -m)" = aarch64 ]; then
    TARGET=rknn
  elif command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    TARGET=cuda
  else
    TARGET=cpu
  fi
  say "Detected target: $TARGET"
fi
case "$TARGET" in cpu|cuda|rknn) ;; *) die "unknown target '$TARGET' (cpu | cuda | rknn)";; esac

if [ ! -x .venv/bin/python ]; then
  say "Creating virtual environment .venv"
  "$PY" -m venv .venv || die "venv creation failed. On Debian/Ubuntu: sudo apt install python3-venv"
fi
VPY=.venv/bin/python
"$VPY" -m pip install --upgrade pip wheel >/dev/null

if "$VPY" -c 'import torch' 2>/dev/null; then
  say "PyTorch already installed: $("$VPY" -c 'import torch; print(torch.__version__)') (kept; delete .venv to reinstall)"
else
  say "Installing PyTorch for $TARGET"
  case "$TARGET" in
    cuda) "$VPY" -m pip install torch --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" ;;
    rknn) "$VPY" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu ;;
    cpu)
      if [ "$(uname -s)" = Darwin ]; then "$VPY" -m pip install torch   # includes Apple GPU (MPS) support
      else "$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu; fi ;;
  esac
fi

say "Installing Python packages"
if [ "$TARGET" = rknn ]; then
  "$VPY" -c 'import sys; sys.exit(sys.version_info[:2] > (3, 12))' \
    || die "rknn-toolkit-lite2 2.3.2 supports Python <= 3.12. Recreate .venv with PYTHON=python3.11 (or use Docker)."
  "$VPY" -m pip install -r requirements-rknn.txt
  command -v ffmpeg >/dev/null || "$VPY" -m pip install imageio-ffmpeg
  say "Downloading the NPU model for this board"
  "$VPY" scripts/download_models.py || die "NPU model download failed (see above)"

  # Rockchip's runtime is proprietary (RKNN SDK License); it is downloaded from Rockchip only after acceptance
  case "$(echo "${RKNN_ACCEPT_LICENSE:-}" | tr 'A-Z' 'a-z')" in yes|true|1) ok=y ;; *) ok= ;; esac
  if [ -z "$ok" ]; then
    echo "The Rockchip NPU runtime (librknnrt.so, rknn-toolkit-lite2 2.3.2) is licensed by Rockchip under the RKNN SDK License:"
    echo "  https://github.com/airockchip/rknn-toolkit2/blob/master/LICENSE"
    printf "Accept it and download the runtime from Rockchip? [y/N] "
    read -r ok || ok=
  fi
  case "$ok" in
    y|Y|yes)
      "$VPY" scripts/fetch_rknn_runtime.py --accept-license --python-only || die "rknn-toolkit-lite2 download failed"
      say "Installing librknnrt.so to /usr/lib (needs sudo)"
      sudo "$VPY" scripts/fetch_rknn_runtime.py --accept-license --lib-only --lib-dir /usr/lib && sudo ldconfig \
        || die "librknnrt.so download failed" ;;
    *) say "Skipped the NPU runtime: the server will run on the CPU. Re-run ./install.sh rknn to add it." ;;
  esac
else
  "$VPY" -m pip install -r requirements.txt
fi

[ -f .env ] || { cp .env.example .env; say "Created .env from .env.example (edit it to configure)"; }

say "Downloading the model (first time only, ~400 MB)"
"$VPY" - <<'EOF'
import os, settings
if not os.path.isdir(settings.MODEL_ID):
    from huggingface_hub import snapshot_download
    snapshot_download(settings.MODEL_ID, allow_patterns=["*.json", "model.safetensors"])
EOF

say "Self-check"
"$VPY" app.py --check || die "self-check failed (see above). docs/DEPLOYMENT.md has a troubleshooting section."

say "Done. Start the server with ./run.sh  ->  http://127.0.0.1:7860 (UI)  /docs (API)"
