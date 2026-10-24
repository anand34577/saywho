# Contributing to saywho

Thanks for helping. saywho tries to stay small and easy to run: a few Python files, one web page, no database.
Please keep changes in that spirit.

## Before you start

- Bugs and ideas go in the [issues](https://github.com/anand34577/saywho/issues). For anything bigger than a bug
  fix, open an issue first so we can agree on the approach before you spend time on it.
- Security problems: see [SECURITY.md](SECURITY.md), not the issue tracker.
- Please don't attach recordings of real people to issues or pull requests unless every speaker has agreed to it.
  `samples/earnest.flac` is public domain and is usually enough to show a problem.

## Development setup

```bash
git clone https://github.com/anand34577/saywho.git
cd saywho
./install.sh            # Windows: install.bat. Creates .venv, installs PyTorch and the model, runs a self-check
./run.sh                # http://127.0.0.1:7860
```

Run the tests before sending a pull request. They use a fake model, so they need no GPU and no model download:

```bash
python -m unittest discover -s tests -v
```

CI runs the same tests on Linux and Windows, checks the shell scripts and compose file, and on changes to the images
builds all three Docker images and diarizes the sample in each of them.

## How it fits together

| Path | What |
|---|---|
| `app.py` | API server and UI host (`python app.py`, `python app.py --check`) |
| `diar_engine.py` | model loading, hardware backends, auto-detection and CPU fallback, Rockchip SoC and core detection |
| `segmentation.py` | probabilities to segments, statistics, RTTM/CSV/SRT/JSON |
| `audio_io.py` | decoding any audio or video file with FFmpeg |
| `timing.py` | per-stage timing breakdown (`timings` in results and logs) |
| `settings.py`, `.env.example` | configuration |
| `static/index.html` | the web app (one file, no build step) |
| `install.sh`, `install.ps1`/`install.bat`, `run.sh`, `run.bat` | native install and start |
| `Dockerfile` (CPU/CUDA), `Dockerfile.rknn` (NPU), `docker-compose.yml` | images |
| `rknn_models/SHA256SUMS`, `scripts/download_models.py` | converted NPU models (release assets) and their downloader |
| `scripts/fetch_rknn_runtime.py`, `docker/rknn-entrypoint.sh` | Rockchip runtime, downloaded from Rockchip after license acceptance |
| `conversion/` | ONNX export, RKNN conversion, validation against the reference model |
| `bench.py`, `scripts/check_npu.sh` | NPU benchmark and driver check |
| `docs/` | guides; `docs/wiki/` is published to the GitHub wiki on every push to `main` |
| `samples/` | the public-domain sample recording |

**The pipeline.** FFmpeg decodes the upload to 16 kHz mono. The model returns, every 10 ms, the probability that each
of its 8 speaker channels is talking. `segmentation.py` turns that into segments by thresholding, merging short gaps
and dropping very short blips. The web app gets the raw probabilities too and repeats the same rules in the browser,
which is why its sliders update instantly.

**On the NPU.** The heavy part (the 31-layer transformer encoder and classifier, 99M parameters) is exported as one
fixed-shape graph of 684 encoder frames (speaker cache 264 + FIFO 40 + chunk 340 + look-ahead 40), padded and masked,
and compiled to FP16 RKNN. Feature extraction and the speaker cache stay the original `transformers` code on the
CPU, so results match the GPU. There are also experimental INT8 models (`RKNN_PRECISION=int8`), which need half
the NPU memory but are not accurate enough yet on real recordings.

## Re-converting the RKNN models

```bash
# Windows or Linux venv with torch + transformers
python conversion/export_onnx.py
# WSL or x86 Linux venv with rknn-toolkit2==2.3.2, onnx==1.16.2, protobuf==4.25.4, numpy==1.26.4, torch>=2.7 (cpu),
#   transformers==5.19.0, librosa==0.11.0, onnxruntime
python conversion/convert_rknn.py --check
# INT8: calibration set from a few minutes of real speech, hybrid quantization, then scoring in the simulator
python conversion/calib_data.py RECORDING [RECORDING ...]
python conversion/convert_rknn.py --quant int8
python conversion/quant_eval.py --platform rk3566
```

Published models are assets of the `rknn-models-v1` release; their checksums live in `rknn_models/SHA256SUMS`.

## Pull requests

- One topic per pull request, with a short description of what changed, why, and how you tested it.
- Commit messages in the imperative, under 72 characters (`Keep the speaker names after re-segmenting`).
- User-visible changes get a line in `CHANGELOG.md`.
- Settings: add new ones to `settings.py`, `.env.example`, `docker-compose.yml`, `docs/DOCKER.md` and the
  [Settings](docs/wiki/Settings.md) wiki page.

By contributing you agree that your contribution is licensed under the [Apache License 2.0](LICENSE).
