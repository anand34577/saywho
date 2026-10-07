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

## Quick start

```bash
git clone https://github.com/anand34577/nemotron-diarization.git
cd nemotron-diarization
```

**Windows** (Python 3.10-3.13 from python.org)
```bat
install.bat
run.bat
```

**Linux / macOS**
```bash
./install.sh
./run.sh
```

**Docker** (any OS)
```bash
docker compose --profile cpu up -d --build      # or: --profile gpu (NVIDIA GPU)
```

**Rockchip board** (NPU): the converted models are release assets, not in git. Download the one for your board
first:
```bash
python3 scripts/download_models.py              # needs `gh auth login` or GITHUB_TOKEN (private repo)
docker compose --profile npu up -d --build      # or natively: ./install.sh rknn (downloads it for you)
```

Then open **http://127.0.0.1:7860** for the UI, or **http://127.0.0.1:7860/docs** for the interactive API docs.

The installers detect your hardware, install the matching PyTorch build, download the model and run a self-check. All
configuration lives in `.env` (see `.env.example`); nothing needs to be set for a local run.

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
| `Dockerfile` (CPU/GPU), `Dockerfile.rknn` (NPU), `docker-compose.yml` | containers |
| `rknn_models/SHA256SUMS`, `scripts/download_models.py` | converted NPU models (release assets) and their downloader |
| `docker/librknnrt.so`, `requirements-rknn.txt` | Rockchip runtime (2.3.2) |
| `conversion/` | ONNX export, RKNN conversion, end-to-end validation against the reference |
| `bench.py`, `scripts/check_npu.sh` | NPU benchmark and driver check |
| `tests/` | API and segmentation tests (`python -m unittest discover -s tests`) |
| `samples/` | test recordings + GPU reference probabilities |

## License

No license file has been added yet. The model weights (`nvidia/Nemotron-3-Diarization`, and the RKNN models converted
from them) are covered by the license on the
[Hugging Face model card](https://huggingface.co/nvidia/Nemotron-3-Diarization).

## Re-converting the RKNN models

```bash
# Windows/Linux venv with torch + transformers
python conversion/export_onnx.py
# WSL / x86 Linux: venv with rknn-toolkit2==2.3.2, onnx==1.16.2, protobuf==4.25.4, numpy==1.26.4, torch>=2.7 (cpu),
#      transformers==5.19.0, librosa==0.11.0, onnxruntime
python conversion/convert_rknn.py --check
```
