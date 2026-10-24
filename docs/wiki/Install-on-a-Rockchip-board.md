# Install on a Rockchip board

saywho can run its heavy part on the NPU (the AI accelerator) of boards with these chips:

| Chip | Example boards | NPU cores |
|---|---|---|
| RK3588 / RK3588S | Orange Pi 5 / 5 Plus, Radxa Rock 5B, Khadas Edge2, NanoPC-T6 | 3 |
| RK3576 | Radxa Rock 4D, Armsom Sige5, Firefly ROC-RK3576 | 2 |
| RK3566 / RK3568 | Radxa Rock 3, Orange Pi 3B, Pine64 Quartz64 | 1 |

On any other ARM board (Raspberry Pi 4/5 and so on) use [Install with Docker](Install-with-Docker) with the `cpu`
image, or [Install on Linux](Install-on-Linux).

**Memory:** at least 2 GB of RAM. Boards with 1 GB are too small for the NPU model at the moment.

## 1. Check the NPU driver

The NPU needs Rockchip's own kernel driver, which comes with the **vendor** kernel of images such as Armbian
("vendor" or "legacy" kernel), Joshua Riek's Ubuntu, or the board maker's own image. Newer mainline kernels use a
different driver that doesn't work with Rockchip's NPU software.

Check it on the board:

```bash
sudo cat /sys/kernel/debug/rknpu/version
```

If it prints something like `RKNPU driver: v0.9.8`, you're set. If it says "No such file or directory", switch to
the vendor kernel (on Armbian: `sudo armbian-config`, then System, then Kernel). Without the driver saywho still
works, just more slowly, on the processor.

## 2. About Rockchip's license

The NPU is driven by Rockchip's runtime (`librknnrt.so`), which is Rockchip's own software under its
[RKNN SDK License](https://github.com/airockchip/rknn-toolkit2/blob/master/LICENSE). saywho doesn't include it. It
downloads it directly from Rockchip, but only after you say you accept that license. Please read it first.

## 3a. Install with Docker (recommended)

Install Docker if the board doesn't have it (`curl -fsSL https://get.docker.com | sudo sh`), then:

```bash
sudo docker run -d --name saywho --privileged -p 7860:7860 --restart unless-stopped \
  -e RKNN_ACCEPT_LICENSE=yes -v rknn-runtime:/opt/rknn ghcr.io/anand34577/saywho:rknn
```

- `RKNN_ACCEPT_LICENSE=yes` is where you accept Rockchip's license.
- `--privileged` lets the container reach the NPU.
- `-v rknn-runtime:/opt/rknn` keeps the downloaded runtime, so it's fetched only once.

The NPU models for all supported chips are inside the image. Open `http://<board-address>:7860` after a minute or
two.

## 3b. Install without Docker

```bash
sudo apt install -y python3 python3-venv git
git clone https://github.com/anand34577/saywho.git
cd saywho
bash install.sh rknn
```

The installer downloads the NPU model for your chip, then asks whether you accept Rockchip's license. Answer `y` to
download the runtime (it asks for your password to place it in `/usr/lib`). Start with `bash run.sh`, and see
[Install on Linux](Install-on-Linux#keep-it-running-in-the-background) to run it as a service.

Python 3.13 isn't supported by Rockchip's runtime yet. If your board has it, use Docker, or install Python 3.11 or
3.12 and run `PYTHON=python3.12 bash install.sh rknn`.

## 4. Check that the NPU is used

The page header should show the NPU, for example **RK3588 NPU (3 cores, mask=all, fp16)**. If it shows the CPU
with "(fallback)", hover over it, or open `http://<board-address>:7860/api/v1/info` and look at `fallbacks`: it
says exactly what's missing.

## Tips

- Even with the NPU, some work stays on the processor. If the board does other jobs too, set `CPU_THREADS=4` (see
  [Settings](Settings)).
- WAV or FLAC files are lighter on the processor than MP3 or video files, because there's less to decode.
- More detail (core masks, pinning to the fast cores, benchmarks) is in
  [DEPLOYMENT.md](https://github.com/anand34577/saywho/blob/main/docs/DEPLOYMENT.md#3-rockchip-npu-boards-rk3566--rk3568--rk3576--rk3588).
