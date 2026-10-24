# Updating and removing

saywho keeps nothing between restarts except your settings, so updating is quick and safe. The
[releases page](https://github.com/anand34577/saywho/releases) lists what changed in each version. The version you
have is shown at `http://localhost:7860/api/v1/info`.

## Update

**Docker:**

```bash
docker pull ghcr.io/anand34577/saywho:cpu        # or :cuda / :rknn
docker rm -f saywho
docker run -d --name saywho -p 7860:7860 --restart unless-stopped ghcr.io/anand34577/saywho:cpu
```

Use the same `docker run` line you used to install, including any `-e` settings. With Compose:
`docker compose pull && docker compose up -d`.

**Installed from a ZIP (Windows or macOS):**

1. Download the new ZIP and extract it, as in the install guide.
2. Copy your `.env` file from the old folder into the new one, if you changed any settings.
3. Run `install.bat` (or `bash install.sh`) in the new folder. It's faster the second time, because the model is
   already downloaded.
4. Delete the old folder once the new one works.

**Installed with git (Linux):**

```bash
cd saywho
git pull
bash install.sh
```

## Remove

**Docker:**

```bash
docker rm -f saywho
docker rmi ghcr.io/anand34577/saywho:cpu         # or :cuda / :rknn
docker volume rm rknn-runtime                    # Rockchip only
```

**Installed with the installer:**

1. Stop saywho (close its window, or Ctrl + C).
2. Delete the saywho folder. Everything saywho installed, Python packages included, is inside it.
3. The speaker model is kept in a shared download cache. To free its space too, delete the folder
   `models--nvidia--Nemotron-3-Diarization` inside:
   - Windows: `C:\Users\<your name>\.cache\huggingface\hub`
   - macOS and Linux: `~/.cache/huggingface/hub`
4. Python itself stays installed. Remove it the usual way (Windows: Settings > Apps) if nothing else needs it.
5. Rockchip only: `sudo rm /usr/lib/librknnrt.so` removes Rockchip's runtime.
