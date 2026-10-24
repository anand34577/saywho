# syntax=docker/dockerfile:1
# saywho (speaker diarization) for CPU or NVIDIA GPU, linux/amd64 and linux/arm64. Rockchip NPU: Dockerfile.rknn.
# Prebuilt images: ghcr.io/anand34577/saywho:cpu and :cuda (see docs/DOCKER.md).
#
#   docker build -t saywho:cpu .
#   docker build -t saywho:cuda --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu129 .
FROM python:3.11-slim-bookworm

# cpu: https://download.pytorch.org/whl/cpu. cuda: .../whl/cu129, the CUDA build with wheels for both amd64 and arm64
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.8.0

LABEL org.opencontainers.image.title="saywho" \
      org.opencontainers.image.description="Who spoke when: speaker diarization web app and REST API (model: nvidia/Nemotron-3-Diarization). Third-party licenses in /licenses." \
      org.opencontainers.image.source="https://github.com/anand34577/saywho" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    DIAR_BACKEND=auto

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install torch==${TORCH_VERSION} --index-url ${TORCH_INDEX}
COPY requirements.txt /tmp/requirements.txt
RUN grep -v imageio-ffmpeg /tmp/requirements.txt > /tmp/req.txt && pip install -r /tmp/req.txt

# model config, processor and weights baked in (starts offline); the model card is kept as its notice of origin
RUN python - <<'EOF'
import os, shutil
from huggingface_hub import snapshot_download
path = snapshot_download("nvidia/Nemotron-3-Diarization", allow_patterns=["*.json", "model.safetensors", "README.md"])
os.makedirs("/licenses", exist_ok=True)
shutil.copy(os.path.join(path, "README.md"), "/licenses/Nemotron-3-Diarization-MODEL_CARD.md")
EOF
ENV HF_HUB_OFFLINE=1

# license inventory of everything installed (pip-licenses runs from a throwaway venv, so it is not in the image)
RUN python -m venv /tmp/pl && /tmp/pl/bin/pip install -q pip-licenses \
 && /tmp/pl/bin/pip-licenses --python /usr/local/bin/python --from=mixed --with-urls --with-license-file \
      --no-license-path --format=plain-vertical > /licenses/python-packages.txt \
 && rm -rf /tmp/pl \
 && dpkg-query -W -f='${Package} ${Version}\n' > /licenses/debian-packages.txt
COPY --chmod=644 LICENSE NOTICE THIRD_PARTY_NOTICES.md licenses/OpenMDW-1.1.txt /licenses/

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY app.py diar_engine.py settings.py segmentation.py audio_io.py timing.py ./
COPY static ./static
COPY samples/earnest.flac samples/README.md ./samples/
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/v1/ready')"
CMD ["python", "app.py"]
