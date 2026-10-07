# Docker images

Prebuilt images are published to the GitHub Container Registry. No login, no clone and no build are needed:

| Image | Platforms | Hardware | Notes |
|---|---|---|---|
| `ghcr.io/anand34577/nemotron-diarization:cpu` | linux/amd64, linux/arm64 | any CPU | also tagged `latest` |
| `ghcr.io/anand34577/nemotron-diarization:cuda` | linux/amd64, linux/arm64 | NVIDIA GPU, CUDA 12.9: amd64 from compute capability 7.0 (Volta, Turing, Ampere, Ada, Hopper, Blackwell); arm64 from 8.0 (A100, GH200, Blackwell) | falls back to the CPU if no GPU is visible |
| `ghcr.io/anand34577/nemotron-diarization:rknn` | linux/arm64 | Rockchip RK3566 / RK3568 / RK3576 / RK3588 NPU | runtime downloaded from Rockchip on first start, see [Rockchip](#rockchip-npu-rknn) |

Every image contains the model (`nvidia/Nemotron-3-Diarization`), so containers start offline. Each image also
serves the web UI at `/` and the REST API at `/api/v1` ([API.md](API.md)) on port **7860**.

**Tags:** `cpu`, `cuda` and `rknn` always point to the newest release. To pin a version, use
`<variant>-<version>` (for example `cpu-2.1.0`) or `<variant>-<major.minor>` (for example `cpu-2.1`).
Docker selects the right architecture automatically.

---

## Quick start

### CPU (any machine)

```bash
docker run -d --name nemotron-diarization -p 7860:7860 --restart unless-stopped \
  ghcr.io/anand34577/nemotron-diarization:cpu
```

Open http://localhost:7860, or test the API:

```bash
curl http://localhost:7860/api/v1/ready
curl -F file=@meeting.mp4 http://localhost:7860/api/v1/diarize
```

### NVIDIA GPU (`cuda`)

Requirements on the host:
1. An NVIDIA driver that supports CUDA 12.9 (version 575 or newer; check with `nvidia-smi`).
2. The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
   Check it with `docker run --rm --gpus all ubuntu nvidia-smi`. On Windows, Docker Desktop with the WSL 2 backend
   only needs the Windows NVIDIA driver.

```bash
docker run -d --name nemotron-diarization --gpus all -p 7860:7860 --restart unless-stopped \
  ghcr.io/anand34577/nemotron-diarization:cuda
```

`curl localhost:7860/api/v1/info` should show `"backend": "cuda"`. If it shows `cpu`, `fallbacks` says why. The usual
cause is a missing `--gpus all` or a missing Container Toolkit.

Supported GPUs follow from the CUDA architectures compiled into PyTorch 2.8 (cu129):
- **amd64:** `sm_70` to `sm_120`, which covers GTX 16xx and RTX 20xx through RTX 50xx, V100, T4, A100, L4, H100
  and B200. Pascal and older cards (GTX 10xx, P100) are not supported; use the `cpu` image for them.
- **arm64:** `sm_80`, `sm_90`, `sm_100` and `sm_120`, for arm64 servers with data-center GPUs (for example Grace
  Hopper or Grace Blackwell).

The arm64 image does **not** run on Jetson boards, which use a different CUDA stack (JetPack); use the `cpu` image
there.

### Rockchip NPU (`rknn`)

On the board (arm64, Armbian/Debian with the vendor kernel; see [Rockchip](#rockchip-npu-rknn)):

```bash
docker run -d --name nemotron-diarization --privileged -p 7860:7860 --restart unless-stopped \
  -e RKNN_ACCEPT_LICENSE=yes -v rknn-runtime:/opt/rknn \
  ghcr.io/anand34577/nemotron-diarization:rknn
```

`RKNN_ACCEPT_LICENSE=yes` means you accept Rockchip's license for the NPU runtime (explained below).

---

## Docker Compose (recommended for servers)

Download the compose file and the settings template. A clone is not needed:

```bash
mkdir nemotron-diarization && cd nemotron-diarization
curl -fsSLO https://raw.githubusercontent.com/anand34577/nemotron-diarization/main/docker-compose.yml
curl -fsSL  https://raw.githubusercontent.com/anand34577/nemotron-diarization/main/.env.example -o .env
```

(While the repository is private, these two URLs need GitHub access. You can copy the two files from someone who
has it.)

Edit `.env`: set `COMPOSE_PROFILES` to `cpu`, `cuda` or `rknn`, plus any settings from the table below. Then:

```bash
docker compose up -d          # pulls the image and starts it
docker compose logs -f        # wait for "... ready on <accelerator>"
docker compose ps             # STATUS shows (healthy) once the model is loaded
```

Without `COMPOSE_PROFILES` in `.env`, name the profile on every command: `docker compose --profile cpu up -d`.

Stop with `docker compose down`. Update with `docker compose pull && docker compose up -d`.

---

## Configuration (environment variables)

Set them with `-e NAME=value` (`docker run`) or in `.env` (compose). All are optional.

### Server

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `7860` | Compose only: host port to publish. With `docker run`, change the left side of `-p 7860:7860` instead. Inside the container the server always listens on 7860 |
| `API_KEY` | *(empty)* | When set, every `/api/v1` call except `health`, `ready` and `info` needs header `X-API-Key: <key>` or `Authorization: Bearer <key>`. The UI asks for it once. Set this whenever the port is reachable from other machines |
| `CORS_ORIGINS` | *(empty)* | Browser origins allowed to call the API from other sites, comma separated, or `*` |
| `ENABLE_UI` | `true` | `false` serves only the API |
| `LOG_LEVEL` | `info` | `debug`, `info`, `warning`, `error` |

### Hardware

| Variable | Default | Meaning |
|---|---|---|
| `DIAR_BACKEND` | `auto` | `auto` (NPU, then CUDA, then CPU), or force `rknn`, `cuda`, `cpu` |
| `DIAR_FALLBACK` | `true` | If the forced backend fails, use the CPU. `false` = report an error instead (`/api/v1/ready` stays 503) |
| `CPU_THREADS` | `0` | PyTorch CPU threads; `0` = all cores |
| `CUDA_DEVICE` | `0` | GPU index when several are visible (or restrict with `--gpus '"device=1"'`) |
| `NPU_CORE_MASK` | `all` | Rockchip: `all`, `auto`, `0`, `1`, `2`, `0_1`, `0_1_2` |
| `RK_PLATFORM` | *(detected)* | Rockchip: force `rk3566`, `rk3568`, `rk3576` or `rk3588` if detection fails |
| `RKNN_ACCEPT_LICENSE` | `no` | Rockchip: `yes` accepts Rockchip's RKNN SDK License and downloads the NPU runtime on start |

### Limits and defaults

| Variable | Default | Meaning |
|---|---|---|
| `MAX_UPLOAD_MB` | `2048` | Largest upload (HTTP 413 above it) |
| `MAX_AUDIO_MINUTES` | `0` | Longest audio, `0` = unlimited. RAM grows by about 4 MB per minute of audio |
| `MAX_QUEUED_JOBS` | `100` | Pending `/api/v1/jobs` before HTTP 429 |
| `JOB_TTL_MINUTES` | `60` | Finished jobs are deleted after this |
| `DEFAULT_THRESHOLD` | `0.5` | Default speech threshold (each request can override it) |
| `DEFAULT_MIN_SEGMENT` | `0.1` | Default minimum segment length, in seconds |
| `DEFAULT_MERGE_GAP` | `0.3` | Default gap merged within one speaker, in seconds |

### Compose only

| Variable | Default | Meaning |
|---|---|---|
| `COMPOSE_PROFILES` | — | `cpu`, `cuda` or `rknn`: which image `docker compose up` runs |
| `IMAGE_VERSION` | *(newest)* | Pin a release, for example `2.1.0` |

The full template with comments is [`.env.example`](../.env.example).

### Ports, volumes and resources

- **Port:** 7860/tcp (UI and API).
- **Volumes:** none are needed for `cpu` and `cuda`, because jobs and uploads are temporary and held in the
  container. For `rknn`, mount a volume at `/opt/rknn` so the Rockchip runtime is downloaded only once.
- **Health:** the images have a `HEALTHCHECK` on `/api/v1/ready`. The status is `starting` until the model is
  loaded (about 10-60 s), then `healthy`.
- **Memory:** about 0.5-1 GB, plus about 4 MB per minute of audio being processed.
- **User:** `cpu` and `cuda` run as the unprivileged user `app` (uid 1000). `rknn` runs as root, because it needs
  `--privileged` for the NPU device.

---

## Rockchip NPU (`rknn`)

### Host requirements

1. A board with RK3566, RK3568, RK3576 or RK3588/RK3588S and a 64-bit OS.
2. The Rockchip **vendor** kernel with the `rknpu` driver (for example the Armbian "vendor" kernel). The mainline
   `rocket` driver does not work with Rockchip's runtime. Check with
   `sudo bash scripts/check_npu.sh` from the repository, or `sudo cat /sys/kernel/debug/rknpu/version`.
3. Docker. The container needs `--privileged`, for the NPU device under `/dev/dri` and the device tree used to detect
   the SoC.

### Why the runtime is downloaded

The NPU runtime (`librknnrt.so` and `rknn-toolkit-lite2`, version 2.3.2) is Rockchip software under the proprietary
[RKNN SDK License](https://github.com/airockchip/rknn-toolkit2/blob/master/LICENSE), so it is not included in the
image. With `RKNN_ACCEPT_LICENSE=yes` you accept that license. On first start, the container then downloads the
runtime directly from Rockchip's GitHub repository and from PyPI, checks the SHA-256, and stores it in `/opt/rknn`.
Keep that path on a volume so the download happens only once.

Without `RKNN_ACCEPT_LICENSE=yes`, or when the board has no internet, the container still starts and runs on the
CPU. The log and `/api/v1/info` → `fallbacks` explain why.

### Check that the NPU is used

```bash
curl localhost:7860/api/v1/info
# "backend": "rknn", "accelerator": "RK3576 NPU (2 cores, mask=0_1)"
```

Every result has a `timings` block (also logged) showing time spent on the NPU against time on the CPU.
[DEPLOYMENT.md](DEPLOYMENT.md#cpu-usage-on-the-board) explains how to reduce CPU load (`CPU_THREADS`, pinning to the
big cores with `--cpuset-cpus 4-7`).

---

## Building the images yourself

From a clone of the repository:

```bash
docker compose --profile cpu  build
docker compose --profile cuda build
python3 scripts/download_models.py all && docker compose --profile rknn build   # models are needed first
```

For other architectures, use `docker buildx build --platform linux/arm64 ...`. The workflow
`.github/workflows/docker.yml` shows the exact release build.

---

## Licenses

Pulling and running an image means using software under several licenses. Your use must comply with all of them:

- This project's code: [Apache-2.0](../LICENSE).
- The model `nvidia/Nemotron-3-Diarization` and its RKNN conversions:
  [OpenMDW-1.1](../licenses/OpenMDW-1.1.txt) (commercial use allowed; keep the license with any redistribution).
- `cuda` image: NVIDIA CUDA and cuDNN runtime libraries, under NVIDIA's license agreements.
- `rknn`: the Rockchip runtime, which you download yourself under the RKNN SDK License (not part of the image).
- Debian, FFmpeg, PyTorch and the Python packages: open-source licenses.

Each image contains the license texts and full package inventories in `/licenses`
(`docker run --rm ghcr.io/anand34577/nemotron-diarization:cpu ls /licenses`). Details:
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `docker compose up` says "no service selected" | Set `COMPOSE_PROFILES=cpu` (or `cuda`/`rknn`) in `.env`, or add `--profile cpu` |
| `pull access denied` / `denied` from ghcr.io | Check the image name and tag. While the package is private, run `docker login ghcr.io` with a token that has `read:packages` |
| `cuda` image runs on the CPU | Add `--gpus all` (compose already reserves the GPUs), install the NVIDIA Container Toolkit, and check `/api/v1/info` → `fallbacks` |
| `could not select device driver "nvidia"` | The NVIDIA Container Toolkit is missing on the host |
| `cuda` fails with "no kernel image is available" | The GPU is older than the image supports. Check with `nvidia-smi --query-gpu=compute_cap --format=csv`; below 7.0 (amd64) or 8.0 (arm64), use the `cpu` image, or a native install with an older PyTorch build |
| `rknn` runs on the CPU | `fallbacks` gives the reason: no `RKNN_ACCEPT_LICENSE=yes`, no `--privileged`, a mainline kernel without `rknpu`, or no internet on first start |
| `exec format error` | Wrong architecture: `rknn` is arm64 only. `cpu` and `cuda` are multi-arch |
| Container `unhealthy` | `docker logs nemotron-diarization`; the model failed to load or the port is taken |
| HTTP 413 on upload | Raise `MAX_UPLOAD_MB`, and the reverse proxy's body size limit |
| Slow on long files behind a proxy | Use the asynchronous `/api/v1/jobs` API, or raise the proxy read timeout |
