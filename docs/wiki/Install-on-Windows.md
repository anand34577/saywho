# Install on Windows

This works on Windows 10 and 11, 64-bit. It takes about 15 minutes, most of it waiting for downloads. You don't need
to type any commands.

There are two ways. **Use the installer** unless you already use Docker Desktop.

## The installer (recommended)

### 1. Install Python

saywho is written in Python, so Windows needs it once.

1. Go to <https://www.python.org/downloads/> and click the yellow **Download Python** button. Any version from
   3.10 to 3.13 works. If the newest version shown is 3.14 or later, scroll down to "Looking for a specific release"
   and pick the newest 3.13.
2. Open the downloaded file.
3. **Important:** on the first screen, tick **Add python.exe to PATH** at the bottom. Then click **Install Now**.
4. When it says "Setup was successful", click **Close**.

Already have Python 3.10 to 3.13? Skip this step.

### 2. Download saywho

1. Open <https://github.com/anand34577/saywho>.
2. Click the green **Code** button, then **Download ZIP**.
3. Open your Downloads folder, right-click `saywho-main.zip` and choose **Extract All...**, then **Extract**.
4. Move the extracted `saywho-main` folder somewhere you'll find it again, for example your Documents folder.

### 3. Run the installer

1. Open the `saywho-main` folder and double-click **`install.bat`**.
2. If Windows says "Windows protected your PC", click **More info**, then **Run anyway**. Windows shows this for
   any downloaded script it doesn't know yet.
3. A black window opens and shows its progress. It:
   - finds Python,
   - checks for an NVIDIA graphics card and downloads the matching version of PyTorch (about 200 MB, or 2.5 GB with
     an NVIDIA card),
   - downloads the speaker model (about 400 MB),
   - and finally runs a self-check on the sample recording.
4. You're done when the window shows:

   ```
   ==> Done. Start the server with run.bat
   ```

   A few lines above, the self-check shows what it found, for example
   `sample: 74 s audio, 3 speakers, 26 segments`.

If it stops with a red **ERROR** line, read that line: it says what's missing. The
[Troubleshooting](Troubleshooting) page covers the common cases.

### 4. Start saywho

Double-click **`run.bat`**. A black window opens (that's saywho running) and your browser opens
<http://127.0.0.1:7860>.

- The first start takes up to a minute while the model loads. The top right of the page shows "Loading model..."
  and then your processor or graphics card.
- **Keep the black window open** while you use saywho. Closing it stops saywho.
- To start saywho again later, double-click `run.bat` again.

Next: [First steps](First-steps).

### Start saywho automatically when you log in (optional)

1. Right-click `run.bat` and choose **Create shortcut**.
2. Press **Windows key + R**, type `shell:startup` and press Enter. A folder opens.
3. Move the shortcut into that folder.

## Docker Desktop

If you already use [Docker Desktop](https://www.docker.com/products/docker-desktop/), open PowerShell and run one
line:

```powershell
# any PC
docker run -d --name saywho -p 7860:7860 --restart unless-stopped ghcr.io/anand34577/saywho:cpu

# PC with an NVIDIA graphics card
docker run -d --name saywho -p 7860:7860 --restart unless-stopped --gpus all ghcr.io/anand34577/saywho:cuda
```

Then open <http://localhost:7860>. More in [Install with Docker](Install-with-Docker).

## Which is faster on my PC?

With an NVIDIA graphics card (GTX 16xx, RTX 20xx or newer), saywho uses it automatically and is about four times
faster than on the processor. The page header shows the card's name when it's in use. Without one, saywho uses the
processor, which is still fast enough for everyday use: about two minutes for an hour of audio.
