# Changelog

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
