# Nemotron-3-Diarization server for CPU (x86_64 / arm64) or NVIDIA GPU. Rockchip NPU boards: Dockerfile.rknn.
#
#   docker compose --profile cpu up -d --build      (or --profile gpu; see docker-compose.yml)
#
#   docker build -t nemotron-diarization:cpu .
#   docker build -t nemotron-diarization:gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 .
FROM python:3.11-slim-bookworm

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.8.0

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

# model config, processor and weights baked into the image: starts offline, no download at runtime
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('nvidia/Nemotron-3-Diarization', allow_patterns=['*.json', 'model.safetensors'])"
ENV HF_HUB_OFFLINE=1

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY app.py diar_engine.py settings.py segmentation.py audio_io.py timing.py ./
COPY static ./static
COPY samples/meeting.wav ./samples/meeting.wav
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/v1/ready')"
CMD ["python", "app.py"]
