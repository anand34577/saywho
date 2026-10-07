"""All runtime configuration, read from environment variables (and an optional `.env` file next to this one).

Every setting has a safe default, so the server runs with no configuration at all. See `.env.example`.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path: str) -> None:
    """Minimal .env reader (KEY=VALUE, # comments, optional quotes). Real environment variables win."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if key.startswith("export "):
                key = key[7:].strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            else:
                value = value.split(" #", 1)[0].strip()
            os.environ.setdefault(key, value)


_load_dotenv(os.environ.get("ENV_FILE", os.path.join(HERE, ".env")))


def _str(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _int(name: str, default: int) -> int:
    value = _str(name)
    try:
        return int(value) if value else default
    except ValueError:
        raise SystemExit(f"config error: {name}={value!r} is not an integer")


def _float(name: str, default: float) -> float:
    value = _str(name)
    try:
        return float(value) if value else default
    except ValueError:
        raise SystemExit(f"config error: {name}={value!r} is not a number")


def _bool(name: str, default: bool) -> bool:
    value = _str(name).lower()
    if not value:
        return default
    return value in ("1", "true", "yes", "on")


# ---- server
HOST = _str("HOST", "127.0.0.1")
PORT = _int("PORT", 7860)
LOG_LEVEL = _str("LOG_LEVEL", "info").lower()
API_KEY = _str("API_KEY")  # empty = no authentication
CORS_ORIGINS = [o.strip() for o in _str("CORS_ORIGINS").split(",") if o.strip()]  # e.g. "*" or "https://a.com,https://b.com"
ENABLE_UI = _bool("ENABLE_UI", True)

# ---- model / hardware
MODEL_ID = _str("MODEL_ID", "nvidia/Nemotron-3-Diarization")  # Hugging Face id or a local directory
DIAR_BACKEND = _str("DIAR_BACKEND", "auto").lower()  # auto | rknn | cuda | mps | cpu | onnx | rknn-sim | torch
DIAR_FALLBACK = _bool("DIAR_FALLBACK", True)  # fall back to the next backend (finally CPU) if the chosen one fails
CPU_THREADS = _int("CPU_THREADS", 0)  # 0 = PyTorch default (all physical cores)
CUDA_DEVICE = _int("CUDA_DEVICE", 0)
ONNX_PATH = _str("ONNX_PATH", os.path.join(HERE, "conversion", "nemotron3_diarization.onnx"))
WARMUP = _bool("WARMUP", True)  # run a 1 s self-test at startup (also validates the backend)

# ---- limits and defaults
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 2048)
MAX_AUDIO_MINUTES = _float("MAX_AUDIO_MINUTES", 0)  # 0 = unlimited
MAX_QUEUED_JOBS = _int("MAX_QUEUED_JOBS", 100)
JOB_TTL_MINUTES = _int("JOB_TTL_MINUTES", 60)  # finished jobs (and their results) are deleted after this
DEFAULT_THRESHOLD = _float("DEFAULT_THRESHOLD", 0.5)
DEFAULT_MIN_SEGMENT = _float("DEFAULT_MIN_SEGMENT", 0.1)
DEFAULT_MERGE_GAP = _float("DEFAULT_MERGE_GAP", 0.3)
