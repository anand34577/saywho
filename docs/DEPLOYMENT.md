# Deployment guide

The same code runs everywhere. At startup it picks the best hardware it finds:

```
Rockchip NPU (rknn)  ->  NVIDIA GPU (cuda)  ->  Apple GPU (mps)  ->  CPU
```

Each candidate gets a 1-second self-test. If one fails (missing driver, wrong kernel, broken CUDA install), the server
logs why and moves on to the next. It always ends on the CPU, so it starts on any machine. `GET /api/v1/info` shows
what was picked (`backend`, `accelerator`) and what was skipped and why (`fallbacks`). The UI header shows the same,
with "(fallback)" when the preferred hardware was unavailable.

## Choose a method

| Where | Easiest | Alternative |
|---|---|---|
| Windows PC (GPU or CPU) | `install.bat` then `run.bat` | Docker Desktop + the `cpu`/`cuda` image |
| Linux server / VM / cloud (GPU or CPU) | Docker image `cuda` or `cpu` ([DOCKER.md](DOCKER.md)) | `./install.sh` then `./run.sh` |
| macOS (Intel / Apple Silicon) | `./install.sh` then `./run.sh` (uses the Apple GPU) | Docker `--profile cpu` (CPU only) |
| Rockchip RK3566/68/76/88 board | Docker image `rknn` (section 3) | `./install.sh rknn` |
| Raspberry Pi 5, Jetson (CPU), Graviton, other arm64 | Docker image `cpu` | `./install.sh cpu` |

Requirements: about 1 GB RAM free (more for very long files), 4 GB disk for CPU installs or 10 GB with CUDA,
Python 3.10-3.13 for native installs, and Docker Compose v2 for Docker installs.

### Get the code

```bash
git clone https://github.com/anand34577/nemotron-diarization.git
cd nemotron-diarization
```

The repository is private: you need read access, and `git` must be logged in, for example through `gh auth login`.
Every command below runs from this folder.

---

## 1. Native install (no Docker)

### Windows

```bat
install.bat          :: auto: NVIDIA GPU if nvidia-smi works, else CPU
install.bat cpu      :: force CPU-only (124 MB PyTorch instead of ~2.5 GB)
run.bat              :: starts the server and opens http://127.0.0.1:7860
```

### Linux / macOS

```bash
chmod +x install.sh run.sh
./install.sh         # auto: Rockchip NPU -> NVIDIA GPU -> CPU   (or: ./install.sh cpu | cuda | rknn)
./run.sh
```

What the installer does: it finds Python, creates `.venv`, installs the PyTorch build that matches the hardware (CUDA,
CPU-only, or the macOS build with Apple GPU support), installs the other packages, creates `.env` from
`.env.example`, downloads the model (~400 MB, once), and runs `python app.py --check`, which diarizes the bundled
sample and prints the backend and speed. Re-running it is safe: it keeps the existing `.venv`, PyTorch and `.env`.

ffmpeg does not need to be installed separately: a system `ffmpeg` on PATH is used if present, otherwise the static
binary from the `imageio-ffmpeg` package.

Installer options: `PYTHON=python3.11 ./install.sh` picks the interpreter (Windows: `$env:PYTHON`).
`TORCH_CUDA=cu128` picks the CUDA wheel; the default `cu126` works with NVIDIA drivers >= 525. For an RTX 50xx
(Blackwell) GPU use `cu128`.

### Manual install (any OS)

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu    # or .../whl/cu126 for NVIDIA
pip install -r requirements.txt
cp .env.example .env
python app.py --check && python app.py
```

PyTorch is installed separately on purpose: `pip install torch` from PyPI gives a CPU-only build on Windows and a
~2.5 GB CUDA build on Linux, and it can silently replace a working CUDA build. On a machine without a GPU,
torch >= 2.7 is required.

---

## 2. Docker

Prebuilt images are published on ghcr.io: `cpu` (amd64 + arm64), `cuda` (amd64 + arm64) and `rknn` (arm64).
**[DOCKER.md](DOCKER.md) is the complete guide**: `docker run` and compose commands, every environment variable,
GPU and NPU host setup, licenses and troubleshooting. In short:

```bash
docker run -d -p 7860:7860 ghcr.io/anand34577/nemotron-diarization:cpu                  # any machine
docker run -d -p 7860:7860 --gpus all ghcr.io/anand34577/nemotron-diarization:cuda      # NVIDIA GPU
```

From a clone, `docker compose --profile cpu|cuda|rknn up -d` pulls the matching image. Add `--build` to build it
locally instead.

### Moving an image to a machine without internet

```bash
docker pull ghcr.io/anand34577/nemotron-diarization:cpu                                  # on a connected machine
docker save ghcr.io/anand34577/nemotron-diarization:cpu | gzip > nemotron-diarization-cpu.tar.gz
gunzip -c nemotron-diarization-cpu.tar.gz | docker load                                  # on the target
```

Add `--platform linux/arm64` to `docker pull` when the target is an arm64 machine.

---

## 3. Rockchip NPU boards (RK3566 / RK3568 / RK3576 / RK3588)

**Check the NPU driver first** (on the host): `sudo bash scripts/check_npu.sh`. The board must run the Armbian
**vendor** kernel (Rockchip `rknpu` driver). Mainline kernels with the `rocket` driver do not work with Rockchip's
runtime (`librknnrt`). Without a working driver the server still runs, on the board's CPU (much slower), and
`/api/v1/info` → `fallbacks` says why.

**Rockchip runtime license.** The NPU runtime (`librknnrt.so`, `rknn-toolkit-lite2` 2.3.2) is Rockchip software
under the proprietary [RKNN SDK License](https://github.com/airockchip/rknn-toolkit2/blob/master/LICENSE). This
project does not ship it. You accept that license and download the runtime directly from Rockchip: in Docker with
`RKNN_ACCEPT_LICENSE=yes`, natively through the prompt in `./install.sh rknn`.

**Option A: prebuilt Docker image** (simplest; the NPU models are included):
```bash
docker run -d --name nemotron-diarization --privileged -p 7860:7860 --restart unless-stopped \
  -e RKNN_ACCEPT_LICENSE=yes -v rknn-runtime:/opt/rknn ghcr.io/anand34577/nemotron-diarization:rknn
```
Compose equivalent: `COMPOSE_PROFILES=rknn` and `RKNN_ACCEPT_LICENSE=yes` in `.env`, then `docker compose up -d`.
See [DOCKER.md](DOCKER.md#rockchip-npu-rknn).

**Option B: build the image yourself.** The converted models (FP16 232-265 MB, INT8 124-157 MB each) are too large for git. They are
assets of the GitHub release `rknn-models-v1`, with checksums in `rknn_models/SHA256SUMS`:
```bash
gh auth login                                   # once; the repository is private (or: export GITHUB_TOKEN=<token>)
python3 scripts/download_models.py              # this board's model (or: rk3566 | rk3568 | rk3576 | rk3588 | all)
                                                # int8 on boards below 2 GB of RAM; force with --int8 / --fp16
docker compose --profile rknn up -d --build
```

**Option C: native, no Docker**: `./install.sh rknn` (Python <= 3.12). It installs the packages, downloads the
model, and, after you accept Rockchip's license, downloads the runtime (into the venv, and `librknnrt.so` into
`/usr/lib` with sudo).

If the model or runtime is missing, the server still starts, on the CPU, and `/api/v1/info` → `fallbacks` says what
to do.

The header shows for example `RK3576 NPU (2 cores, mask=0_1)`.

### NPU cores

| SoC | Model file | Cores | Default mask |
|---|---|---|---|
| RK3576 | `nemotron3_diarization_rk3576.rknn` | 2 | `0_1` (both cores) |
| RK3588/RK3588S | `nemotron3_diarization_rk3588.rknn` | 3 | `0_1_2` (all three) |
| RK3566/RK3568 | `nemotron3_diarization_rk3566.rknn` | 1 | `auto` |

The model processes a file in sequential ~27 s chunks (each depends on the speaker cache of the previous one), so the
cores are used by splitting every chunk's graph across them (`core_mask`), not by running files in parallel. Override
with `NPU_CORE_MASK` (`all`, `auto`, `0`, `1`, `2`, `0_1`, `0_1_2`) and the SoC with `RK_PLATFORM`.

### CPU usage on the board

Even with the NPU, some work stays on the CPU. Before the first NPU call, the whole file is decoded (ffmpeg) and
turned into a spectrogram. Each NPU call then has some CPU work around it (speaker cache, padding). The `timings` in
every result and the log line per file show how long each part took and how many cores it kept busy:

```
long.mp3 (31.1 min): decode 3.24s (1.0 cores) | features 2.12s (3.4 cores) | inference 77.05s (1.2 cores) [accelerator 75.17s in 69 calls, CPU-side 1.88s] | ...
```

(illustrative; read your own numbers with `docker compose logs -f`)

To keep the CPU load down:
- **`CPU_THREADS=4`** in `.env`. By default PyTorch uses every core. On RK3576/RK3588 that includes the slower
  Cortex-A53/A55 cores, and the faster cores end up waiting for them. With 4 threads the remaining cores stay free
  for the rest of the system.
- **Pin to the big cores.** Check the layout first: `lscpu`, or
  `cat /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq` (the big cores have the highest frequency). Then add
  `cpuset: "4-7"` (for example) to the `rknn` service in `docker-compose.yml` (`--cpuset-cpus 4-7` with
  `docker run`), or start natively with
  `taskset -c 4-7 ./run.sh`.
- **Upload WAV or FLAC** when you control the source. MP3/AAC decoding is skipped; only the spectrogram remains.

Hardware video decoding (RKMPP / ffmpeg-rockchip) does not help. MPP accelerates video codecs and JPEG only; audio
(MP3, AAC, Opus) is always decoded on the CPU. The video stream of a video file is never decoded anyway (ffmpeg
runs with `-vn`).

### Verify on the board

The test recordings are not in the image. From a clone, copy them in first:

```bash
docker cp samples nemotron-diarization:/app/samples
# correctness vs the GPU reference (expect identical segments, 3 speakers)
docker exec nemotron-diarization python conversion/validate.py samples/meeting.wav --backend rknn --reference samples/meeting_ref.npy
docker exec nemotron-diarization python conversion/validate.py samples/long.wav    --backend rknn --reference samples/long_ref.npy
# speed for each core mask (stop the web app first to free the NPU, or run it while idle)
docker exec nemotron-diarization python bench.py
# per-core NPU load while a file is processing (host)
sudo watch -n0.5 cat /sys/kernel/debug/rknpu/load
```

---

## 4. Configuration

Everything is set through environment variables or a `.env` file in the project folder. Real environment variables
override `.env`. `.env.example` documents every option. The most useful ones:

| Variable | Default | Purpose |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `7860` | Listen address. Use `HOST=0.0.0.0` to accept connections from other machines (Docker always does) |
| `API_KEY` | empty | Require a key on API calls |
| `CORS_ORIGINS` | empty | Let browser apps on other origins call the API (`*` or a comma-separated list) |
| `ENABLE_UI` | `true` | `false` serves only the API |
| `DIAR_BACKEND` | `auto` | `auto`, `rknn`, `cuda`, `mps`, `cpu`, `onnx` |
| `DIAR_FALLBACK` | `true` | Fall back to the CPU if the forced backend fails. `false` makes the server report an error instead |
| `CPU_THREADS` | all cores | Limit CPU threads (for example on a shared server) |
| `MAX_UPLOAD_MB` / `MAX_AUDIO_MINUTES` | `2048` / unlimited | Request limits |
| `MODEL_ID` | `nvidia/Nemotron-3-Diarization` | Hub id or a local folder (offline installs) |
| `HF_HUB_OFFLINE` | unset | `1` = never contact Hugging Face |

## 5. Running in production

### As a Linux service (native install)

`/etc/systemd/system/diarization.service`:

```ini
[Unit]
Description=Nemotron Diarization API
After=network-online.target

[Service]
User=diar
WorkingDirectory=/opt/nemotron-diarization
ExecStart=/opt/nemotron-diarization/.venv/bin/python app.py
Environment=HF_HUB_OFFLINE=1
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

`sudo systemctl daemon-reload && sudo systemctl enable --now diarization`, then `journalctl -u diarization -f`.
Run `install.sh` as the service user, so the model ends up in that user's cache.

On Windows, run `run.bat` from Task Scheduler at log-on, or wrap `.venv\Scripts\python.exe app.py` with a service
manager such as NSSM.

### Behind a reverse proxy (HTTPS, public access)

Keep `HOST=127.0.0.1` and put nginx (or Caddy, Traefik) in front. Raise the upload size and timeouts:

```nginx
server {
    listen 443 ssl;
    server_name diarization.example.com;
    # ssl_certificate ...;

    client_max_body_size 2g;          # match MAX_UPLOAD_MB
    proxy_read_timeout   3600s;       # long synchronous /diarize calls (or use /jobs)
    proxy_send_timeout   3600s;
    proxy_request_buffering off;      # stream uploads straight to the app

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Set `API_KEY` whenever the server is reachable from outside your machine.

### Health checks

- Liveness: `GET /api/v1/health`, which answers `200` as soon as the process runs
- Readiness: `GET /api/v1/ready`, which answers `200` only after the model has loaded (allow 10 s to 3 min, longer
  on first start while the model downloads)

The Docker images already define a `HEALTHCHECK` on `/ready`. Kubernetes:

```yaml
livenessProbe:  {httpGet: {path: /api/v1/health, port: 7860}, periodSeconds: 30}
readinessProbe: {httpGet: {path: /api/v1/ready,  port: 7860}, periodSeconds: 10, failureThreshold: 30}
```

### Capacity

One model instance processes one file at a time; concurrent requests queue. The CPU container uses about 0.5 GB of
RAM (measured after processing a 3-minute file), plus about 4 MB per minute of audio being processed. To serve more load, run several instances (one per GPU, for example with
`CUDA_DEVICE=1`) behind a load balancer. `/jobs` state is per instance, so use sticky sessions or a shared result
store if you scale out with jobs.

### Measured speed (52 s meeting sample)

| Hardware | Backend | Speed |
|---|---|---|
| NVIDIA GTX 1650 (4 GB) | cuda | ~110x real time |
| AMD Ryzen 5 5600H, 6 cores | cpu | ~60x real time |
| Same CPU | onnx | ~42x real time (CPU PyTorch is faster; ONNX is for validation) |
| RK3566 / RK3576 / RK3588 NPU | rknn | not measured yet: run `bench.py` on the board |

On a GPU, a very long file that runs out of GPU memory is retried on the CPU automatically rather than failing.

## 6. Updating

```bash
git pull
./install.sh                                   # native (Windows: install.bat); keeps .venv, PyTorch and .env
docker compose pull && docker compose up -d   # Docker, prebuilt images (add --profile <p> without COMPOSE_PROFILES)
```

`CHANGELOG.md` lists what changed in each version.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| UI header says "(fallback)" | The preferred hardware failed its check. Hover over it, or see `/api/v1/info` → `fallbacks` |
| `torch ... is too old for a machine without a usable GPU` | Install torch >= 2.7: `pip install -U torch --index-url https://download.pytorch.org/whl/cpu` |
| `/ready` stays 503, `/info` shows `error` | The model failed to load. The usual cause is no internet on first start: run the installer once while online, or copy `~/.cache/huggingface` over and set `HF_HUB_OFFLINE=1` |
| NVIDIA GPU present but `backend: cpu` | CPU-only PyTorch is installed. Delete `.venv` and run `install.bat cuda` / `./install.sh cuda` |
| `Could not decode audio ...` | The file is damaged or has no audio track. Without ffmpeg only WAV/FLAC/OGG/MP3 work: `pip install imageio-ffmpeg` |
| `413` on upload | Raise `MAX_UPLOAD_MB` (and nginx `client_max_body_size`) |
| Proxy returns 504 on long files | Use `/api/v1/jobs`, or raise `proxy_read_timeout` |
| UI can't be reached from another machine | Set `HOST=0.0.0.0` (native) and open the port in the firewall |
| Rockchip: `init_runtime failed` / `rknn: ... fallbacks` | Vendor kernel and `rknpu` driver needed: `sudo bash scripts/check_npu.sh` |
| `uvicorn ... Unknown level: 'INFO'` | Start with `python app.py`, not with the `uvicorn` CLI (rknn-toolkit-lite2 changes logging level names) |

Automated tests (no GPU or model needed): `python -m unittest discover -s tests -v`.
