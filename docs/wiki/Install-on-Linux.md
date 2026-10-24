# Install on Linux

Works on any 64-bit Linux (Ubuntu, Debian, Fedora, Arch, Raspberry Pi OS 64-bit and others), on x86 PCs and ARM
boards. If the machine already runs Docker, [Install with Docker](Install-with-Docker) is the shortest route.

## 1. Get Python and git

saywho needs Python 3.10 to 3.13. Most current distributions have it. On Ubuntu or Debian:

```bash
sudo apt update
sudo apt install -y python3 python3-venv git
python3 --version        # should print 3.10, 3.11, 3.12 or 3.13
```

Fedora: `sudo dnf install -y python3 git`. Arch: `sudo pacman -S python git`.

If your distribution's Python is newer than 3.13, install 3.12 or 3.13 alongside it and tell the installer to use it:
`PYTHON=python3.12 bash install.sh`.

## 2. Download and install

```bash
git clone https://github.com/anand34577/saywho.git
cd saywho
bash install.sh
```

The installer looks for an NVIDIA graphics card (with a working `nvidia-smi`) and installs the matching build of
PyTorch, or the smaller processor-only build. It downloads the speaker model (about 400 MB) and ends with a
self-check on the sample recording:

```
sample:      74 s audio, 3 speakers, 26 segments, ...
OK
==> Done. Start the server with ./run.sh
```

To choose yourself: `bash install.sh cpu` (smallest download) or `bash install.sh cuda` (NVIDIA).

## 3. Start saywho

```bash
bash run.sh
```

Open <http://127.0.0.1:7860>. Stop it with Ctrl + C.

By default saywho only accepts connections from the same machine. To open it from other devices, see
[Use it from other devices](Use-it-from-other-devices).

Next: [First steps](First-steps).

## Keep it running in the background

To start saywho at boot and restart it if it stops, run it as a systemd service. Replace `/home/you/saywho` with the
folder you cloned into, and `you` with your user name:

```bash
sudo tee /etc/systemd/system/saywho.service >/dev/null <<'EOF'
[Unit]
Description=saywho
After=network-online.target

[Service]
User=you
WorkingDirectory=/home/you/saywho
ExecStart=/home/you/saywho/.venv/bin/python app.py
Environment=HF_HUB_OFFLINE=1
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now saywho
journalctl -u saywho -f        # watch the log; Ctrl + C to stop watching
```

More server topics (HTTPS, reverse proxy, health checks) are in
[DEPLOYMENT.md](https://github.com/anand34577/saywho/blob/main/docs/DEPLOYMENT.md).
