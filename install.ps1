# One-step install for Windows. Run from this folder:
#
#   .\install.bat            auto-detect: NVIDIA GPU -> CPU
#   .\install.bat cpu        force a CPU-only install (smallest download)
#   .\install.bat cuda       force the NVIDIA CUDA build of PyTorch
#
# Env: $env:PYTHON = "C:\Path\python.exe" to pick the interpreter, $env:TORCH_CUDA = "cu128" (default cu126).
# Re-running is safe: the existing .venv, PyTorch and .env are kept.
param([ValidateSet("auto", "cpu", "cuda")][string]$Target = "auto")
# "Continue": in Windows PowerShell 5.1, "Stop" turns any stderr output of python/pip into a fatal error;
# failures are detected through $LASTEXITCODE instead
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

function Say($msg) { Write-Host "==> $msg" -ForegroundColor Green }
function Die($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }
function Run($exe, [string[]]$argv) { & $exe @argv; if ($LASTEXITCODE -ne 0) { Die "command failed: $exe $($argv -join ' ')" } }

# ---- find Python 3.10-3.13
$py = $env:PYTHON
if (-not $py) {
    foreach ($v in "3.12", "3.11", "3.13", "3.10") {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            $p = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $p) { $py = $p.Trim(); break }
        }
    }
}
if (-not $py -and (Get-Command python -ErrorAction SilentlyContinue)) { $py = (Get-Command python).Source }
if (-not $py) { Die "Python 3.10-3.13 not found. Install it from https://www.python.org/downloads/ (tick 'Add to PATH')." }
& $py -c "import sys; sys.exit(not ((3, 10) <= sys.version_info[:2] <= (3, 13)))"
if ($LASTEXITCODE -ne 0) { Die "Python 3.10-3.13 required, found: $(& $py --version 2>&1). Set `$env:PYTHON." }

if ($Target -eq "auto") {
    $Target = "cpu"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi -L *> $null
        if ($LASTEXITCODE -eq 0) { $Target = "cuda" }
    }
    Say "Detected target: $Target"
}

$vpy = ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    Say "Creating virtual environment .venv"
    Run $py @("-m", "venv", ".venv")
}
& $vpy -m pip install --upgrade pip wheel *> $null

& $vpy -c "import torch" 2>$null
if ($LASTEXITCODE -eq 0) {
    Say "PyTorch already installed: $(& $vpy -c 'import torch; print(torch.__version__)') (kept; delete .venv to reinstall)"
} else {
    Say "Installing PyTorch for $Target"
    $cuda = if ($env:TORCH_CUDA) { $env:TORCH_CUDA } else { "cu126" }
    $index = if ($Target -eq "cuda") { "https://download.pytorch.org/whl/$cuda" } else { "https://download.pytorch.org/whl/cpu" }
    Run $vpy @("-m", "pip", "install", "torch", "--index-url", $index)
}

Say "Installing Python packages"
Run $vpy @("-m", "pip", "install", "-r", "requirements.txt")

if (-not (Test-Path .env)) { Copy-Item .env.example .env; Say "Created .env from .env.example (edit it to configure)" }

Say "Downloading the model (first time only, ~400 MB)"
Run $vpy @("-c", "import os, settings; from huggingface_hub import snapshot_download; os.path.isdir(settings.MODEL_ID) or snapshot_download(settings.MODEL_ID, allow_patterns=['*.json', 'model.safetensors'])")

Say "Self-check"
& $vpy app.py --check
if ($LASTEXITCODE -ne 0) { Die "self-check failed (see above). docs/DEPLOYMENT.md has a troubleshooting section." }

Say "Done. Start the server with run.bat  ->  http://127.0.0.1:7860 (UI)  /docs (API)"
