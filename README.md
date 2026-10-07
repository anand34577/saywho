# Nemotron Diarization: who spoke when, on any hardware

Speaker diarization with `nvidia/Nemotron-3-Diarization`, served as a **REST API** with a **web UI**. Upload a video or
audio file and get speaker segments, per-speaker statistics, overlaps and turn-taking, plus RTTM/CSV/SRT/JSON exports.

It runs on whatever hardware is available and picks the fastest option automatically:

| Hardware | Backend | Notes |
|---|---|---|
| Rockchip RK3566 / RK3568 / RK3576 / RK3588 NPU | `rknn` | converted FP16 models in `rknn_models/` |
| NVIDIA GPU | `cuda` | any CUDA GPU (tested on a 4 GB GTX 1650) |
| Apple Silicon GPU | `mps` | macOS |
| Any CPU (x86_64, arm64) | `cpu` | ~60x real time on a 6-core laptop CPU |

If the preferred hardware fails at startup (no driver, wrong kernel, broken CUDA), the server logs why and **falls
back to the CPU** instead of failing. CPU output matches the GPU reference exactly.

## Quick start with Docker (prebuilt images)

```bash
# any machine (x86_64 or arm64)
docker run -d --name nemotron-diarization -p 7860:7860 ghcr.io/anand34577/nemotron-diarization:cpu

# NVIDIA GPU (needs the NVIDIA Container Toolkit)
docker run -d --name nemotron-diarization -p 7860:7860 --gpus all ghcr.io/anand34577/nemotron-diarization:cuda

# Rockchip RK3566/68/76/88 board; RKNN_ACCEPT_LICENSE=yes accepts Rockchip's runtime license (see docs/DOCKER.md)
docker run -d --name nemotron-diarization -p 7860:7860 --privileged \
  -e RKNN_ACCEPT_LICENSE=yes -v rknn-runtime:/opt/rknn ghcr.io/anand34577/nemotron-diarization:rknn
```

Then open **http://localhost:7860** for the UI, or **http://localhost:7860/docs** for the interactive API docs.
**[docs/DOCKER.md](docs/DOCKER.md)** covers compose, every environment variable, GPU and NPU host setup, and
troubleshooting.

## Quick start without Docker

```bash
git clone https://github.com/anand34577/nemotron-diarization.git
cd nemotron-diarization
```

| OS | Install | Start |
|---|---|---|
| Windows (Python 3.10-3.13 from python.org) | `install.bat` | `run.bat` |
| Linux / macOS | `./install.sh` | `./run.sh` |
| Rockchip board | `./install.sh rknn` | `./run.sh` |

The installers detect your hardware, install the matching PyTorch build, download the model and run a self-check.
On a Rockchip board, `./install.sh rknn` also downloads the NPU model and, once you accept Rockchip's license, the NPU
runtime. All configuration lives in `.env` (see `.env.example`); nothing needs to be set for a local run. Details:
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## API

```bash
curl -F file=@meeting.mp4 http://127.0.0.1:7860/api/v1/diarize                         # JSON
curl -F file=@meeting.mp4 -F format=rttm -o meeting.rttm http://127.0.0.1:7860/api/v1/diarize
```

```json
{
  "duration": 51.9, "num_speakers": 3, "accelerator": "NVIDIA GeForce GTX 1650 (CUDA)", "real_time_factor": 109.6,
  "speakers": [{"label": "Speaker 1", "talk_time": 13.7, "share_of_speech": 0.384, "segments": 7, ...}],
  "segments": [{"speaker": "Speaker 1", "start": 0.07, "end": 1.32, "duration": 1.25, "overlaps_with": []}, ...],
  "summary": {"speech_time": 35.7, "overlap_time": 0.0, "speaker_turns": 7, "interruptions": 0, ...},
  "transitions": {...}
}
```

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/diarize` | Upload and wait; JSON or an `rttm`/`csv`/`srt` file |
| `POST /api/v1/jobs`, `GET /api/v1/jobs/{id}`, `GET /api/v1/jobs/{id}/result` | Async jobs for long files |
| `GET /api/v1/health`, `/ready`, `/info` | Liveness, readiness, backend and limits |

Parameters: `threshold`, `min_segment`, `merge_gap`, `include_probabilities`, `format`. Optional API key auth
(`API_KEY`), CORS, and upload/duration limits. **Full reference: [docs/API.md](docs/API.md)**. Python client:
[examples/client.py](examples/client.py).

## Documentation

- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**: every install method (Windows, Linux, macOS, Docker CPU/GPU/NPU,
  Rockchip boards), configuration, systemd, nginx/HTTPS, health checks, offline installs, troubleshooting
- **[docs/DOCKER.md](docs/DOCKER.md)**: prebuilt images (`cpu`, `cuda`, `rknn`), `docker run` and compose, every
  environment variable, licenses inside the images
- **[docs/API.md](docs/API.md)**: endpoints, parameters, response fields, errors, curl/Python/JavaScript examples
- **[.env.example](.env.example)**: every setting with its default

## How it works

- Audio is decoded to 16 kHz mono by ffmpeg (a system install, or the bundled `imageio-ffmpeg` binary). The model
  outputs per-speaker activity probabilities every 10 ms. Segments are derived by thresholding, gap merging and
  minimum-length filtering, with the same rules on the server (API) and in the browser (UI, where the sliders
  re-segment instantly).
- On the NPU, the heavy part (31-layer transformer encoder + classifier, 99M params) is exported as one fixed-shape
  graph (684 encoder frames = speaker cache 264 + FIFO 40 + chunk 340 + look-ahead 40), padded and masked, and
  compiled to FP16 RKNN. Feature extraction, feature stacking and the Arrival-Order Speaker Cache stay the original
  transformers code on the CPU, so results match the GPU.
- RKNN validation (x86 simulator, FP16, full pipeline): identical segments to the GPU on the 52 s sample for
  rk3566/rk3576/rk3588; on the 186 s sample 72/72 segments match, and 0.02 % of 10 ms frames differ at a boundary.

## Files

| Path | What |
|---|---|
| `app.py` | API server + UI host (`python app.py`, `python app.py --check`) |
| `diar_engine.py` | model loading, backends, auto-detection and fallback, Rockchip SoC/core detection |
| `segmentation.py` | probabilities → segments, statistics, RTTM/CSV/SRT/JSON |
| `audio_io.py` | decoding any audio/video file |
| `timing.py` | per-stage timing breakdown (`timings` in results and logs) |
| `settings.py`, `.env.example` | configuration |
| `static/index.html` | web UI |
| `install.sh`, `install.ps1`/`install.bat`, `run.sh`, `run.bat` | native install and start |
| `Dockerfile` (CPU/CUDA), `Dockerfile.rknn` (NPU), `docker-compose.yml`, `.github/workflows/docker.yml` | images and their release build |
| `rknn_models/SHA256SUMS`, `scripts/download_models.py` | converted NPU models (release assets) and their downloader |
| `scripts/fetch_rknn_runtime.py`, `docker/rknn-entrypoint.sh`, `requirements-rknn.txt` | Rockchip runtime 2.3.2, downloaded from Rockchip after license acceptance |
| `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, `licenses/` | licenses and notices |
| `conversion/` | ONNX export, RKNN conversion, end-to-end validation against the reference |
| `bench.py`, `scripts/check_npu.sh` | NPU benchmark and driver check |
| `tests/` | API and segmentation tests (`python -m unittest discover -s tests`) |
| `samples/` | test recordings + GPU reference probabilities |

## License and terms of use

- **This project's code:** [Apache License 2.0](LICENSE). Copyright 2026 Anand. See [NOTICE](NOTICE).
- **The model** `nvidia/Nemotron-3-Diarization` (NVIDIA), and the RKNN models converted from it:
  [OpenMDW License 1.1](licenses/OpenMDW-1.1.txt). Commercial and non-commercial use are allowed; if you
  redistribute the model, include that license.
- **Rockchip NPU runtime:** Rockchip's proprietary
  [RKNN SDK License](https://github.com/airockchip/rknn-toolkit2/blob/master/LICENSE). It is not distributed here;
  you download it from Rockchip after accepting that license.
- **Docker images:** they also contain Debian, FFmpeg, PyTorch, Python packages and, in the `cuda` image, NVIDIA
  CUDA libraries, each under its own license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `/licenses`
  inside each image.

**Responsible use.** Speaker diarization processes recordings of people's voices, which is personal data in many
jurisdictions (for example under the GDPR). Only process recordings you are allowed to process: get consent where
the law requires it, and follow the recording and data-protection laws that apply to you. The model labels
speakers by order of appearance ("Speaker 1", "Speaker 2"); it does not identify who a person is, and should not be
used to try. Results can contain errors (for example with recordings of more than eight speakers, the model's
maximum, noisy audio or very short turns), so review them before relying on them for decisions about people. The software and the model are
provided "as is", without warranty of any kind (Apache-2.0 sections 7-8; OpenMDW-1.1).

NVIDIA, Nemotron and CUDA are trademarks of NVIDIA Corporation; Rockchip and RKNN are trademarks of Rockchip
Electronics Co., Ltd. This project is independent and not endorsed by either company.

## Re-converting the RKNN models

```bash
# Windows/Linux venv with torch + transformers
python conversion/export_onnx.py
# WSL / x86 Linux: venv with rknn-toolkit2==2.3.2, onnx==1.16.2, protobuf==4.25.4, numpy==1.26.4, torch>=2.7 (cpu),
#      transformers==5.19.0, librosa==0.11.0, onnxruntime
python conversion/convert_rknn.py --check
```
