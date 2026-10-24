# Troubleshooting

Two places tell you most of what's going on:

- **The window where saywho runs** (or `docker logs saywho` for Docker). Problems show up as lines with `WARNING` or
  `ERROR`.
- **`http://localhost:7860/api/v1/info`** in your browser. It shows the version, the hardware in use, and under
  `fallbacks` why faster hardware was skipped.

## Installing

| What you see | What to do |
|---|---|
| `Python 3.10-3.13 not found` | Install Python from <https://www.python.org/downloads/>. On Windows, tick **Add python.exe to PATH** in the installer, then run `install.bat` again |
| `Python 3.10-3.13 required, found 3.14` (or 3.9) | Install Python 3.12 or 3.13 as well. Windows picks it up by itself; on Linux and macOS run `PYTHON=python3.12 bash install.sh` |
| "Windows protected your PC" | Click **More info**, then **Run anyway**. Windows shows this for any downloaded script |
| `venv creation failed` (Linux) | `sudo apt install python3-venv`, then run the installer again |
| The download stops or times out | Run the installer again. It picks up where it stopped and keeps what's already installed |
| `self-check failed` | Read the lines above it. Most often the model couldn't be downloaded: check the internet connection and run the installer again |

## Starting

| What you see | What to do |
|---|---|
| The page doesn't open | saywho may still be starting: wait a minute and reload. Check that its window is still open (or `docker ps` shows it) |
| `.venv not found - run install.bat first` | The installer hasn't finished successfully in this folder yet. Run it again |
| The header says "Loading model..." for minutes | The first start on a new computer downloads the model. Watch the saywho window for progress |
| "Model failed to load" | Usually no internet on the very first start. Connect and restart saywho once; after that it works offline |
| `address already in use` / port 7860 is taken | Another program (or a second saywho) uses port 7860. Close it, or set `PORT=7861` in `.env` ([Settings](Settings)) |
| The page works on the computer but not from my phone | See [Use it from other devices](Use-it-from-other-devices) |

## Results

| What you see | What to do |
|---|---|
| "Could not decode audio" | The file is damaged or has no sound track. Try playing it in VLC; if it plays, convert it to MP3 or WAV and try again |
| HTTP 413, "larger than MAX_UPLOAD_MB" | Raise `MAX_UPLOAD_MB` in [Settings](Settings). Behind a web server, raise its upload limit too |
| Quiet people are missing | Lower **Speech threshold** to 0.3 or 0.4 |
| Noise or music shows up as speech | Raise **Speech threshold** to 0.6 or 0.7, and **Min segment** to 0.3 |
| One person is split into two speakers, or two people share one | That's the model. It happens most with similar voices, poor microphones, phone calls and more than 8 speakers. Rename speakers to merge them by eye |
| It's much slower than expected | Check the header: if it shows `CPU` with "(fallback)" although you have a graphics card or NPU, see below |
| Long files fail behind a web server with a "504" error | Use the job API for long files, or raise the server's timeout (see [DEPLOYMENT.md](https://github.com/anand34577/saywho/blob/main/docs/DEPLOYMENT.md)) |

## Hardware

| What you see | What to do |
|---|---|
| NVIDIA card, but the header shows `CPU` | The processor-only version of PyTorch was installed. Delete the `.venv` folder inside the saywho folder and run `install.bat cuda` (or `bash install.sh cuda`) |
| Docker `cuda` image runs on the CPU | Add `--gpus all` to `docker run`, and install the NVIDIA Container Toolkit on Linux. `fallbacks` in `/api/v1/info` says which part is missing |
| `could not select device driver "nvidia"` | The NVIDIA Container Toolkit isn't installed on the Docker host |
| "no kernel image is available" (cuda image) | The graphics card is too old for the image (older than GTX 16xx). Use the `cpu` image |
| Rockchip board runs on the CPU | `fallbacks` says why: `RKNN_ACCEPT_LICENSE=yes` missing, `--privileged` missing, no NPU driver (see [Install on a Rockchip board](Install-on-a-Rockchip-board#1-check-the-npu-driver)), or no internet on first start |
| `exec format error` (Docker) | The `rknn` image only runs on 64-bit ARM boards. Use `cpu` or `cuda` elsewhere |

## Still stuck?

[Open an issue](https://github.com/anand34577/saywho/issues/new/choose). Include what you did, what you expected,
and the output of `http://localhost:7860/api/v1/info`. Please don't attach recordings of real people unless they
agreed to it.
