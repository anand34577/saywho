# Install on macOS

Works on Apple Silicon (M1 and newer) and Intel Macs, macOS 12 or newer. On Apple Silicon, saywho uses the Mac's
graphics processor automatically. It takes about 15 minutes.

You'll type a few commands into **Terminal**. Each one is shown below: copy it, paste it into Terminal, press
Return, and wait until the next prompt appears.

## 1. Install Python

The Python that comes with macOS is too old.

1. Go to <https://www.python.org/downloads/macos/> and download the **macOS 64-bit universal2 installer** of the
   newest Python 3.13 (3.10 to 3.13 all work).
2. Open the downloaded `.pkg` and click through the installer.

## 2. Download saywho

1. Open <https://github.com/anand34577/saywho>, click the green **Code** button, then **Download ZIP**.
2. Double-click `saywho-main.zip` in your Downloads folder. A `saywho-main` folder appears.
3. Move it somewhere you'll find again, for example your Documents folder.

## 3. Run the installer

1. Open **Terminal** (press Cmd + Space, type `Terminal`, press Return).
2. Type `cd ` (with a space after it), drag the `saywho-main` folder from Finder into the Terminal window, and press
   Return.
3. Run:

   ```bash
   bash install.sh
   ```

   It creates a private Python environment inside the folder, installs PyTorch and the other packages, downloads
   the speaker model (about 400 MB) and runs a self-check. It finishes with:

   ```
   ==> Done. Start the server with ./run.sh
   ```

If it stops with a red **ERROR** line, read that line: it says what's missing. See also
[Troubleshooting](Troubleshooting).

## 4. Start saywho

In the same Terminal window:

```bash
bash run.sh
```

Then open <http://127.0.0.1:7860> in Safari, Chrome or Firefox. The first start takes up to a minute while the model
loads.

**Keep the Terminal window open** while you use saywho. To stop it, press Ctrl + C in that window. To start it again
later, open Terminal, `cd` into the folder as in step 3, and run `bash run.sh`.

Next: [First steps](First-steps).

## Docker on a Mac

The `cpu` Docker image runs on Docker Desktop for Mac, but Docker can't use the Mac's graphics processor, so the
installer above is faster on Apple Silicon. See [Install with Docker](Install-with-Docker) if you still prefer it.
