# Third-party notices

This project's own code is under the Apache License 2.0 (`LICENSE`). It uses and, in its Docker images and
releases, redistributes the components below, each under its own terms. Inside every image, these files are in
`/licenses`:

| File | What |
|---|---|
| `/licenses/LICENSE`, `/licenses/NOTICE`, `/licenses/THIRD_PARTY_NOTICES.md` | this project |
| `/licenses/OpenMDW-1.1.txt`, `/licenses/Nemotron-3-Diarization-MODEL_CARD.md` | model license and model card (notice of origin) |
| `/licenses/python-packages.txt` | every installed Python package with its license and full license text, generated at build time |
| `/licenses/debian-packages.txt` | every installed Debian package and its version (license texts: `/usr/share/doc/<package>/copyright`) |

This file describes licenses for convenience. It is not legal advice; the license texts are authoritative.

## Model: nvidia/Nemotron-3-Diarization

- **Source:** https://huggingface.co/nvidia/Nemotron-3-Diarization (NVIDIA Corporation)
- **License:** OpenMDW License Agreement, version 1.1. Full text: `licenses/OpenMDW-1.1.txt`, or
  https://openmdw.ai/license/1-1/. The model card states it is ready for commercial and non-commercial use.
- **Where it is redistributed:** the weights and configuration (`config.json`, `processor_config.json`,
  `model.safetensors`) are baked into every Docker image. The RKNN conversions (`nemotron3_diarization_*.rknn`) are
  in the `rknn` image and in the `rknn-models-v1` GitHub release.
- **How the obligations are met:** OpenMDW-1.1 requires every distribution to include a copy of the agreement and
  the model's copyright and origin notices. The agreement and the model card (its notice of origin; it contains no
  separate copyright lines) ship with every image in `/licenses`, and the agreement is attached to the model
  release. The license places no restrictions on the outputs (diarization results).
- **Disclaimer:** the model comes "AS IS", without warranty, and you are responsible for any third-party rights and
  consents your use requires (OpenMDW-1.1, final paragraphs).

## Sample recording: `samples/earnest.flac`

A 74-second excerpt of the LibriVox dramatic reading of *The Importance of Being Earnest* by Oscar Wilde
(https://archive.org/details/importanceofbeingearnest_1205_librivox), marked with the Public Domain Mark 1.0. It is
in this repository and in every Docker image. Details: `samples/README.md`.

## Rockchip RKNN runtime (`rknn` image and Rockchip native installs only)

- **Components:** `librknnrt.so` 2.3.2 and the `rknn-toolkit-lite2` 2.3.2 Python package, from
  https://github.com/airockchip/rknn-toolkit2 and https://pypi.org/project/rknn-toolkit-lite2/.
- **License:** Rockchip's proprietary **RKNN SDK License**
  (https://github.com/airockchip/rknn-toolkit2/blob/master/LICENSE). Among other terms: no reverse engineering,
  notices must be kept, export-control compliance is required, and the license ends automatically on breach.
- **Not redistributed by this project.** The images and this repository do not contain these files. When you set
  `RKNN_ACCEPT_LICENSE=yes` (Docker) or confirm the prompt in `./install.sh rknn` (native), you accept Rockchip's
  license. `scripts/fetch_rknn_runtime.py` then downloads the files directly from Rockchip's GitHub repository and
  from PyPI, and checks their SHA-256.
- **The `.rknn` model files** were produced from the model above with Rockchip's RKNN-Toolkit2 2.3.2. They are
  distributed under OpenMDW-1.1, as works made for running on Rockchip hardware (RKNN SDK License, section 1.1).

## NVIDIA CUDA libraries (`cuda` image only)

PyTorch's CUDA build installs NVIDIA runtime libraries as Python packages (`nvidia-cublas-*`, `nvidia-cudnn-*`,
`nvidia-cuda-runtime-*`, `nvidia-nccl-*` and others). They are distributed under NVIDIA's license agreements,
mainly the CUDA Toolkit EULA (https://docs.nvidia.com/cuda/eula/index.html) and the cuDNN Software License Agreement
(https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html), which allow these runtime components to be
redistributed with applications. Each package's license text is in `/licenses/python-packages.txt`. By using the
`cuda` image you agree to those terms. A host NVIDIA driver is required; it is not part of the image.

## Open-source software in the images

| Component | License | Notes |
|---|---|---|
| Debian 12 (bookworm) base system, via the `python:3.11-slim` image | various (mostly GPL, LGPL, BSD, MIT) | list in `/licenses/debian-packages.txt` |
| FFmpeg (Debian `ffmpeg` package and its libraries) | GPL-2.0-or-later / LGPL-2.1-or-later (Debian build) | see below |
| Python 3.11 | PSF License | |
| PyTorch | BSD-3-Clause | https://github.com/pytorch/pytorch |
| Hugging Face Transformers, huggingface_hub | Apache-2.0 | |
| FastAPI, Starlette, Uvicorn, NumPy, SciPy, librosa, soundfile and others | MIT / BSD / ISC and similar | full list and texts in `/licenses/python-packages.txt` |

**Source code for GPL/LGPL components.** The Debian packages in the images are unmodified, and their exact
versions are listed in `/licenses/debian-packages.txt`. The corresponding source code is available from Debian at
https://snapshot.debian.org/ and https://sources.debian.org/, or with `apt-get source <package>=<version>`. If you
cannot get it that way, contact the maintainer (https://github.com/anand34577) and the source will be provided for at
least three years after the image was published.

**Native installs** (`install.sh`, `install.bat`) download every package from its original source (PyPI, the PyTorch
index, Hugging Face) under that package's own license. `imageio-ffmpeg` bundles an FFmpeg binary under the GPL; see
https://github.com/imageio/imageio-ffmpeg.

## Trademarks

NVIDIA, Nemotron and CUDA are trademarks of NVIDIA Corporation. Rockchip and RKNN are trademarks of Rockchip
Electronics Co., Ltd. saywho is an independent project and is not affiliated with or endorsed by NVIDIA or Rockchip. The model's name
is used only to say which model saywho runs.
