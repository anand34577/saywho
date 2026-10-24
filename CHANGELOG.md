# Changelog

## 2.2.1 - 2026-10-24

### Fixed
- The INT8 RKNN models are no longer chosen automatically. On a real 11.8 min recording on an RK3576 the INT8
  model found 7 speakers where FP16 found the correct 2, although it matched the reference in the simulator on the
  test samples (including one held out of calibration). FP16 is the default again on every board, including those
  below 2 GB of RAM; INT8 stays available as an experiment with `RKNN_PRECISION=int8` and `download_models.py --int8`.

## 2.2.0 - 2026-10-08

### Added
- INT8 RKNN models for RK3566/RK3568, RK3576 and RK3588 (`nemotron3_diarization_<soc>_int8.rknn`, 124-157 MB
  instead of 232-265 MB), so the NPU backend fits boards with 1 GB of RAM. Hybrid quantization: int8 weights and
  activations, with the attention score path kept in fp16 (plain int8 breaks it: 60 % of frames decided
  differently). In the simulator, every segment of both samples matches the GPU reference within 0.1 s, with ~1 % of
  10 ms frames decided differently. `RKNN_PRECISION=auto|int8|fp16`; `auto` picks int8 below 2 GB of RAM.
- `conversion/convert_rknn.py --quant int8`, `conversion/calib_data.py` (calibration set from real audio) and
  `conversion/quant_eval.py` (end-to-end simulator scoring against the reference).
- `scripts/download_models.py --int8 | --fp16`; without either, int8 is downloaded on boards below 2 GB of RAM.

### Changed
- The rknn and onnx backends no longer load the transformer encoder weights into PyTorch; only the feature
  embedder runs on the CPU.

### Fixed
- `rknn` Docker image: the NPU backend failed with "Can not find dynamic library" because rknn-toolkit-lite2 only
  looks for `/usr/lib/librknnrt.so`; the entrypoint now links the downloaded runtime there.

## 2.1.0 - 2026-10-07

### Added
- Prebuilt Docker images on ghcr.io: `cpu` (amd64 + arm64), `cuda` (amd64 + arm64, CUDA 12.9) and `rknn` (arm64,
  RKNN models included), built and smoke-tested by `.github/workflows/docker.yml` for every release.
- `docs/DOCKER.md`: running every image, compose, every environment variable, GPU/NPU host setup, troubleshooting.
- Licensing: the project is now under Apache-2.0 (`LICENSE`, `NOTICE`); `THIRD_PARTY_NOTICES.md` and `licenses/`
  cover the model (OpenMDW-1.1) and bundled software. Every image has license texts and full package inventories
  in `/licenses`.
- Responsible-use notes in the README.

### Changed
- The Rockchip NPU runtime is no longer bundled (Rockchip's RKNN SDK License). It is downloaded from Rockchip after
  you accept that license: `RKNN_ACCEPT_LICENSE=yes` in Docker, or the prompt in `./install.sh rknn`
  (`scripts/fetch_rknn_runtime.py`). Without it the server runs on the CPU.
- Compose profiles are renamed to match the image tags: `cpu`, `cuda`, `rknn` (were `cpu`, `gpu`, `npu`).
  `docker compose up` now pulls the prebuilt image; add `--build` to build locally.
- The CUDA image uses the CUDA 12.9 build of PyTorch (`cu129`), the only one with arm64 wheels.
- Sample recordings are no longer copied into the images.

## 2.0.0 - 2026-10-07

### Added
- Runs on any machine: picks the Rockchip NPU, NVIDIA GPU, Apple GPU or CPU automatically, and falls back to the
  CPU when the preferred hardware fails its startup check. `/api/v1/info` and the UI show what was used and why.
- REST API under `/api/v1`: synchronous `diarize`, asynchronous `jobs` for long files, `health`, `ready`, `info`.
  Results include segments, per-speaker statistics, a conversation summary and turn transitions. Exports to RTTM,
  CSV and SRT. Interactive docs at `/docs`.
- Per-stage timing breakdown (decode, features, inference, NPU vs CPU-side time, average busy cores) in every result,
  in the log and in the UI.
- Optional API key, CORS, upload-size and audio-length limits. Configuration through `.env` (see `.env.example`).
- One-step installers (`install.bat`, `install.sh`) that pick the right PyTorch build, download the model and run
  `python app.py --check`.
- Docker images for CPU (amd64 and arm64) and NVIDIA GPU, next to the Rockchip image, selected through compose
  profiles.
- `scripts/download_models.py`: fetches the RKNN models from the release and verifies their checksums.
- A very long file that runs out of GPU memory is processed on the CPU instead of failing.
- Works without a system ffmpeg (bundled `imageio-ffmpeg`).

### Changed
- `docker compose up` needs a profile now: `--profile cpu|gpu|npu`, or `COMPOSE_PROFILES` in `.env`.
- The Rockchip Dockerfile is now `Dockerfile.rknn`; `Dockerfile` builds the CPU/GPU image.
- The RKNN models are no longer stored with the code; download them with `scripts/download_models.py`.
- `torch` is no longer listed in `requirements.txt`. Install the build that matches your hardware, or let the
  installers do it.

### Compatibility
- `GET /api/info` and `POST /api/diarize` still work. `/api/diarize` returns a superset of the old response.

## 1.0.0

- Web UI for diarizing a video or audio file with `nvidia/Nemotron-3-Diarization`, on a CUDA GPU or the NPU of
  RK3566/RK3568/RK3576/RK3588 boards. Statistics, timeline, turn-taking, and RTTM/CSV/SRT/JSON downloads.
