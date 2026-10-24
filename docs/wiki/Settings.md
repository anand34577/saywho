# Settings

saywho works without any settings. Change them only if you need to.

## Where settings go

**Installed with `install.bat` or `install.sh`:** the installer creates a file called `.env` in the saywho folder.
Open it with a text editor (Notepad on Windows, TextEdit on a Mac). Each setting is one line, `NAME=value`, with an
explanation above it. Save the file, then restart saywho (close the window and run `run.bat` or `run.sh` again).

On Windows, the easiest way is to right-click `.env` and choose **Open with > Notepad**. From inside Notepad, use
**File > Open** and pick **All files (\*.\*)** at the bottom right, because Notepad lists only `.txt` files by
default.

**Docker with Compose:** the same `.env` file, next to `docker-compose.yml`. Run `docker compose up -d` after
changing it.

**Docker with `docker run`:** add `-e NAME=value` for each setting, for example
`docker run -d --name saywho -p 7860:7860 -e API_KEY=my-long-secret ghcr.io/anand34577/saywho:cpu`. To change a
setting later, remove the container (`docker rm -f saywho`) and run the command again with the new value.

## The settings you're most likely to want

| Setting | Default | What it does |
|---|---|---|
| `PORT` | `7860` | The port in the address, `http://localhost:7860`. Change it if something else already uses 7860 |
| `HOST` | `127.0.0.1` | `127.0.0.1` means only this computer can open saywho. `0.0.0.0` lets other devices in, see [Use it from other devices](Use-it-from-other-devices). Docker always uses `0.0.0.0` |
| `API_KEY` | empty | A password for saywho. When set, the web page asks for it once, and programs must send it. Use a long random value |
| `MAX_UPLOAD_MB` | `2048` | The largest file accepted, in MB |
| `MAX_AUDIO_MINUTES` | `0` (no limit) | The longest recording accepted, in minutes. Each minute of audio needs about 4 MB of memory while it's processed |
| `CPU_THREADS` | `0` (all) | How many processor threads saywho may use. Set it lower to keep the computer responsive for other work |
| `DEFAULT_THRESHOLD`, `DEFAULT_MIN_SEGMENT`, `DEFAULT_MERGE_GAP` | `0.5`, `0.1`, `0.3` | Starting values of the three boxes on the page, and the defaults for the API |

To make a random key for `API_KEY`, run this in the saywho folder (Windows: `.venv\Scripts\python`):

```bash
.venv/bin/python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Hardware settings

| Setting | Default | What it does |
|---|---|---|
| `DIAR_BACKEND` | `auto` | Which hardware to use. `auto` picks the fastest: Rockchip NPU, then NVIDIA GPU, then Apple GPU, then the processor. Or force one: `rknn`, `cuda`, `mps`, `cpu` |
| `DIAR_FALLBACK` | `true` | If the chosen hardware fails, use the processor instead. `false` makes saywho report the error instead |
| `CUDA_DEVICE` | `0` | Which NVIDIA card to use when there are several (0 is the first) |
| `NPU_CORE_MASK` | `all` | Rockchip: which NPU cores to use |
| `RK_PLATFORM` | detected | Rockchip: set `rk3566`, `rk3568`, `rk3576` or `rk3588` if the chip isn't detected |
| `RKNN_ACCEPT_LICENSE` | `no` | Rockchip Docker image: `yes` accepts Rockchip's license and downloads the NPU runtime |
| `RKNN_PRECISION` | `fp16` | Rockchip: `int8` is an experiment that uses half the NPU memory but is not accurate enough yet |

## Everything else

| Setting | Default | What it does |
|---|---|---|
| `ENABLE_UI` | `true` | `false` turns off the web page and leaves only the API |
| `CORS_ORIGINS` | empty | Lets web pages on other sites call the API: a comma-separated list of addresses, or `*` for any |
| `LOG_LEVEL` | `info` | How much saywho writes to its log: `debug`, `info`, `warning` or `error` |
| `MAX_QUEUED_JOBS` | `100` | How many files can wait in the API's job queue |
| `JOB_TTL_MINUTES` | `60` | How long finished job results are kept |
| `MODEL_ID` | `nvidia/Nemotron-3-Diarization` | The model, or a folder holding a downloaded copy (for computers without internet) |
| `HF_HUB_OFFLINE` | not set | `1` stops saywho from ever contacting Hugging Face once the model is downloaded |

The full list with comments is in
[.env.example](https://github.com/anand34577/saywho/blob/main/.env.example).
