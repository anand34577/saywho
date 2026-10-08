"""Nemotron-3-Diarization server: REST API + web UI.

Upload any audio/video file; ffmpeg extracts 16 kHz mono audio, the model returns per-speaker activity
probabilities, and the server turns them into segments, per-speaker statistics and RTTM/CSV/SRT/JSON exports.

    python app.py                      # http://127.0.0.1:7860  (UI at /, API docs at /docs)

The hardware backend is chosen by DIAR_BACKEND (auto | rknn | cuda | mps | cpu | onnx): auto uses the Rockchip NPU
when a converted model for the detected SoC exists, else an NVIDIA GPU, else an Apple GPU, else the CPU. If the
chosen backend fails at startup the server falls back to the CPU (DIAR_FALLBACK=false to disable). All settings:
.env.example.
"""
import hmac
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from enum import Enum

import numpy as np
from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import settings
from audio_io import SAMPLE_RATE, AudioDecodeError, ffmpeg_path, load_audio
from segmentation import analyze, export
from timing import stage_clock

__version__ = "2.2.0"
HERE = os.path.dirname(os.path.abspath(__file__))
MAX_SENT_FRAMES = 150_000  # cap on probability frames returned when include_probabilities=true
LEVELS = {"critical": 50, "error": 40, "warning": 30, "info": 20, "debug": 10}

# numeric levels: rknn-toolkit-lite2 rewrites logging's level names on import, so names can't be relied on
logging.basicConfig(level=LEVELS.get(settings.LOG_LEVEL, 20), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("diarization")


class Format(str, Enum):
    json = "json"
    rttm = "rttm"
    csv = "csv"
    srt = "srt"


# ---------------------------------------------------------------- engine (loaded in the background)


class EngineState:
    def __init__(self):
        self.engine = None
        self.status = "loading"  # loading | ready | error
        self.error = None
        self.ready = threading.Event()

    def load(self):
        t0 = time.perf_counter()
        try:
            from diar_engine import DiarizationEngine

            self.engine = DiarizationEngine(settings.DIAR_BACKEND, onnx_path=settings.ONNX_PATH,
                                            fallback=settings.DIAR_FALLBACK, cpu_threads=settings.CPU_THREADS)
            self.status = "ready"
            info = self.engine.info()
            log.info("%s ready on %s (backend: %s) in %.1f s", info["model"], info["accelerator"], info["backend"],
                     time.perf_counter() - t0)
            for reason in info["fallbacks"]:
                log.warning("fallback: %s", reason)
        except BaseException as e:  # noqa: BLE001 - SystemExit from version checks included
            self.status, self.error = "error", f"{type(e).__name__}: {e}"
            log.exception("model failed to load")
        finally:
            self.ready.set()

    def require(self):
        if self.status == "loading":
            raise HTTPException(503, "Model is still loading, retry shortly.", headers={"Retry-After": "10"})
        if self.status == "error":
            raise HTTPException(503, f"Model failed to load: {self.error}")
        return self.engine


state = EngineState()


# ---------------------------------------------------------------- pipeline


def save_upload(file: UploadFile, workdir: str) -> str:
    """Stream the upload to disk, enforcing MAX_UPLOAD_MB."""
    ext = os.path.splitext(file.filename or "")[1][:16]
    path = os.path.join(workdir, "input" + ext)
    limit, size = settings.MAX_UPLOAD_MB * 1024 * 1024, 0
    with open(path, "wb") as f:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if limit and size > limit:
                raise HTTPException(413, f"File is larger than MAX_UPLOAD_MB={settings.MAX_UPLOAD_MB}.")
            f.write(chunk)
    if size == 0:
        raise HTTPException(400, "Uploaded file is empty.")
    return path


def downsample(probs: np.ndarray, factor: int) -> np.ndarray:
    """Max-pools frames so short speech bursts survive the reduction."""
    if factor <= 1:
        return probs
    t = (len(probs) + factor - 1) // factor * factor
    padded = np.pad(probs, ((0, t - len(probs)), (0, 0)))
    return padded.reshape(-1, factor, probs.shape[1]).max(axis=1)


def diarize_file(path: str, filename: str | None, threshold: float, min_segment: float, merge_gap: float,
                 include_probabilities: bool) -> dict:
    engine = state.require()
    t0 = time.perf_counter()
    clock = stage_clock()
    try:
        audio = load_audio(path)
    except AudioDecodeError as e:
        raise HTTPException(400, str(e))
    decode = clock.stop()
    duration = len(audio) / SAMPLE_RATE
    if duration < 0.5:
        raise HTTPException(400, "Audio is shorter than 0.5 s.")
    if settings.MAX_AUDIO_MINUTES and duration > settings.MAX_AUDIO_MINUTES * 60:
        raise HTTPException(413, f"Audio is {duration / 60:.1f} min, longer than MAX_AUDIO_MINUTES={settings.MAX_AUDIO_MINUTES:g}.")

    probs, timings = engine.run(audio)
    timings = {"decode": decode, **timings}
    inference_time = timings["inference"]["wall"]
    del audio
    clock = stage_clock()
    analysis = analyze(probs, engine.frame_sec, duration, threshold, min_segment, merge_gap)
    timings["segmentation"] = clock.stop()
    result = {
        "filename": filename,
        "model": settings.MODEL_ID,
        "backend": engine.backend,
        "accelerator": engine.accelerator,
        "duration": round(duration, 3),
        "inference_time": round(inference_time, 3),
        "real_time_factor": round(duration / max(inference_time, 1e-6), 1),
        "parameters": {"threshold": threshold, "min_segment": min_segment, "merge_gap": merge_gap},
        **analysis,
    }
    if include_probabilities:
        # keep only channels the model ever activates; the UI re-segments from these when its sliders change
        channels = [c for c in range(probs.shape[1]) if probs[:, c].max() >= 0.05]
        probs = probs[:, channels] if channels else probs[:, :0]
        factor = max(1, int(np.ceil(len(probs) / MAX_SENT_FRAMES)))
        result["channels"] = channels
        result["activity"] = {"frame_sec": engine.frame_sec * factor, "probs": np.round(downsample(probs, factor), 3).tolist()}
    result["processing_time"] = timings["total"] = round(time.perf_counter() - t0, 3)
    result["timings"] = timings
    log.info("%s (%.1f min): %s", filename, duration / 60, format_timings(timings))
    return result


def format_timings(t: dict) -> str:
    """One log line, e.g. 'decode 2.1s (1.0 cores) | features 1.3s (5.2 cores) | inference 20.0s ...'."""
    def stage(name):
        s = t[name]
        return f"{name} {s['wall']:.2f}s" + (f" ({s['cores']} cores)" if s.get("cores") is not None else "")

    parts = [stage("decode"), stage("features"), stage("inference")]
    if "accelerator" in t:
        parts[-1] += f" [accelerator {t['accelerator']['wall']:.2f}s in {t['accelerator']['calls']} calls, " \
                     f"CPU-side {t['cpu_side']['wall']:.2f}s]"
    if t["queue_wait"] >= 0.05:
        parts.append(f"queue wait {t['queue_wait']:.2f}s")
    parts += [stage("segmentation"), f"total {t['total']:.2f}s"]
    return " | ".join(parts)


def render(result: dict, fmt: Format) -> Response:
    if fmt == Format.json:
        return JSONResponse(result)
    text, media_type = export(result, fmt.value)
    base = os.path.splitext(result.get("filename") or "audio")[0] or "audio"
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in base)
    return Response(text, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="{safe}.{fmt.value}"'})


# ---------------------------------------------------------------- async jobs


class Jobs:
    """In-memory job queue, processed one at a time (the model is serialised anyway)."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diar-job")

    def purge(self):
        cutoff = time.time() - settings.JOB_TTL_MINUTES * 60
        with self.lock:
            for jid in [j for j, job in self.jobs.items() if job["finished_at"] and job["finished_at"] < cutoff]:
                del self.jobs[jid]

    def submit(self, path, workdir, filename, params) -> dict:
        self.purge()
        with self.lock:
            if sum(j["status"] in ("queued", "running") for j in self.jobs.values()) >= settings.MAX_QUEUED_JOBS:
                raise HTTPException(429, f"Too many pending jobs (MAX_QUEUED_JOBS={settings.MAX_QUEUED_JOBS}).")
            jid = uuid.uuid4().hex
            job = {"id": jid, "status": "queued", "filename": filename, "parameters": params, "created_at": time.time(),
                   "started_at": None, "finished_at": None, "error": None, "result": None, "workdir": workdir}
            self.jobs[jid] = job
        job["future"] = self.pool.submit(self._run, job, path)
        return job

    def _run(self, job, path):
        job["status"], job["started_at"] = "running", time.time()
        try:
            state.ready.wait()
            p = job["parameters"]
            job["result"] = diarize_file(path, job["filename"], p["threshold"], p["min_segment"], p["merge_gap"],
                                         p["include_probabilities"])
            job["status"] = "done"
        except HTTPException as e:
            job["status"], job["error"] = "error", e.detail
        except Exception as e:  # noqa: BLE001
            log.exception("job %s failed", job["id"])
            job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
        finally:
            job["finished_at"] = time.time()
            shutil.rmtree(job["workdir"], ignore_errors=True)

    def get(self, jid) -> dict:
        self.purge()
        job = self.jobs.get(jid)
        if not job:
            raise HTTPException(404, "Job not found (unknown id, or expired after JOB_TTL_MINUTES).")
        return job

    def public(self, job, with_result=True) -> dict:
        out = {k: job[k] for k in ("id", "status", "filename", "parameters", "created_at", "started_at", "finished_at", "error")}
        if job["status"] == "queued":
            with self.lock:
                out["queue_position"] = sum(j["status"] == "queued" and j["created_at"] < job["created_at"] for j in self.jobs.values())
        out["status_url"] = f"/api/v1/jobs/{job['id']}"
        out["result_url"] = f"/api/v1/jobs/{job['id']}/result"
        if with_result and job["status"] == "done":
            out["result"] = job["result"]
        return out

    def delete(self, jid):
        job = self.get(jid)
        if job["status"] == "running":
            raise HTTPException(409, "Job is running and cannot be cancelled; delete it after it finishes.")
        if job["status"] == "queued" and not job["future"].cancel():
            raise HTTPException(409, "Job already started.")
        shutil.rmtree(job["workdir"], ignore_errors=True)
        with self.lock:
            self.jobs.pop(jid, None)


jobs = Jobs()


# ---------------------------------------------------------------- HTTP


def require_api_key(x_api_key: str | None = Header(None), authorization: str | None = Header(None)):
    if not settings.API_KEY:
        return
    token = x_api_key
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token or not hmac.compare_digest(token.encode(), settings.API_KEY.encode()):
        raise HTTPException(401, "Missing or invalid API key (send header 'X-API-Key: <key>' or 'Authorization: Bearer <key>').",
                            headers={"WWW-Authenticate": "Bearer"})


def diarize_params(
    threshold: float = Form(settings.DEFAULT_THRESHOLD, ge=0.01, le=0.99, description="Speech probability threshold"),
    min_segment: float = Form(settings.DEFAULT_MIN_SEGMENT, ge=0, le=60, description="Drop segments shorter than this (s)"),
    merge_gap: float = Form(settings.DEFAULT_MERGE_GAP, ge=0, le=60, description="Merge same-speaker gaps shorter than this (s)"),
    include_probabilities: bool = Form(False, description="Also return raw per-frame speaker probabilities (large)"),
) -> dict:
    return {"threshold": threshold, "min_segment": min_segment, "merge_gap": merge_gap,
            "include_probabilities": include_probabilities}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Nemotron Diarization server %s starting; loading %s (backend=%s)", __version__, settings.MODEL_ID, settings.DIAR_BACKEND)
    if not ffmpeg_path():
        log.warning("ffmpeg not found: only WAV/FLAC/OGG/MP3 can be decoded (pip install imageio-ffmpeg, or install ffmpeg)")
    if settings.API_KEY:
        log.info("API key authentication enabled")
    threading.Thread(target=state.load, name="model-loader", daemon=True).start()
    yield
    jobs.pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="Nemotron Diarization API",
    version=__version__,
    description="Speaker diarization (who spoke when) with nvidia/Nemotron-3-Diarization on Rockchip NPU, "
                "NVIDIA GPU, Apple GPU or CPU. See docs/API.md for examples.",
    lifespan=lifespan,
)
if settings.CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"],
                       allow_credentials="*" not in settings.CORS_ORIGINS)

public = APIRouter(prefix="/api/v1", tags=["service"])
api = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])


@public.get("/health", summary="Liveness: the process is up")
def health():
    return {"status": "ok", "model_status": state.status}


@public.get("/ready", summary="Readiness: the model is loaded (503 until then)")
def ready():
    if state.status != "ready":
        return JSONResponse({"status": state.status, "error": state.error}, status_code=503)
    return {"status": "ready"}


@public.get("/info", summary="Model, hardware backend, limits and defaults")
def info():
    out = {
        "version": __version__,
        "status": state.status,
        "error": state.error,
        "auth_required": bool(settings.API_KEY),
        "ffmpeg": bool(ffmpeg_path()),
        "limits": {"max_upload_mb": settings.MAX_UPLOAD_MB, "max_audio_minutes": settings.MAX_AUDIO_MINUTES or None},
        "defaults": {"threshold": settings.DEFAULT_THRESHOLD, "min_segment": settings.DEFAULT_MIN_SEGMENT,
                     "merge_gap": settings.DEFAULT_MERGE_GAP},
        "model": settings.MODEL_ID,
        "backend": None,
        "accelerator": "loading model…" if state.status == "loading" else None,
    }
    if state.engine:
        out.update(state.engine.info())
    return out


@api.post("/diarize", tags=["diarization"], summary="Diarize a file and wait for the result")
def diarize(file: UploadFile = File(..., description="Any audio or video file"), params: dict = Depends(diarize_params),
            format: Format = Form(Format.json, description="json, or a file export: rttm | csv | srt")):
    state.require()
    workdir = tempfile.mkdtemp(prefix="diar_")
    try:
        path = save_upload(file, workdir)
        return render(diarize_file(path, file.filename, **params), format)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


@api.post("/jobs", tags=["jobs"], status_code=202, summary="Queue a file; poll the job for the result")
def create_job(file: UploadFile = File(..., description="Any audio or video file"), params: dict = Depends(diarize_params)):
    if state.status == "error":
        state.require()
    workdir = tempfile.mkdtemp(prefix="diar_job_")
    try:
        path = save_upload(file, workdir)
        return jobs.public(jobs.submit(path, workdir, file.filename, params))
    except BaseException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise


@api.get("/jobs", tags=["jobs"], summary="List jobs (without results)")
def list_jobs():
    jobs.purge()
    return {"jobs": [jobs.public(j, with_result=False) for j in sorted(jobs.jobs.values(), key=lambda j: j["created_at"])]}


@api.get("/jobs/{job_id}", tags=["jobs"], summary="Job status (includes the JSON result when done)")
def get_job(job_id: str):
    return jobs.public(jobs.get(job_id))


@api.get("/jobs/{job_id}/result", tags=["jobs"], summary="Job result as json | rttm | csv | srt")
def get_job_result(job_id: str, format: Format = Format.json):
    job = jobs.get(job_id)
    if job["status"] == "error":
        raise HTTPException(422, f"Job failed: {job['error']}")
    if job["status"] != "done":
        raise HTTPException(409, f"Job is {job['status']}; poll /api/v1/jobs/{job_id} until status is 'done'.",
                            headers={"Retry-After": "5"})
    return render(job["result"], format)


@api.delete("/jobs/{job_id}", tags=["jobs"], status_code=204, summary="Delete a finished job or cancel a queued one")
def delete_job(job_id: str):
    jobs.delete(job_id)
    return Response(status_code=204)


app.include_router(public)
app.include_router(api)


# legacy (v1.x) routes, kept so existing scripts keep working
@app.get("/api/info", include_in_schema=False)
def legacy_info():
    return info()


@app.post("/api/diarize", include_in_schema=False, dependencies=[Depends(require_api_key)])
def legacy_diarize(file: UploadFile = File(...)):
    params = {"threshold": settings.DEFAULT_THRESHOLD, "min_segment": settings.DEFAULT_MIN_SEGMENT,
              "merge_gap": settings.DEFAULT_MERGE_GAP, "include_probabilities": True}
    return diarize(file, params, Format.json)


if settings.ENABLE_UI:
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(HERE, "static", "index.html"))

    app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


def check() -> int:
    """`python app.py --check`: load the model with the current settings, diarize the bundled sample, report."""
    state.load()
    if state.status != "ready":
        print(f"FAILED: {state.error}")
        return 1
    info = state.engine.info()
    print(f"backend:     {info['backend']} (requested: {info['requested_backend']})")
    print(f"accelerator: {info['accelerator']}")
    for reason in info["fallbacks"]:
        print(f"fallback:    {reason}")
    print(f"ffmpeg:      {ffmpeg_path() or 'not found (only WAV/FLAC/OGG/MP3 supported)'}")
    sample = os.path.join(HERE, "samples", "meeting.wav")
    if os.path.exists(sample):
        r = diarize_file(sample, "meeting.wav", settings.DEFAULT_THRESHOLD, settings.DEFAULT_MIN_SEGMENT,
                         settings.DEFAULT_MERGE_GAP, False)
        print(f"sample:      {r['duration']:.0f} s audio, {r['num_speakers']} speakers, {len(r['segments'])} segments, "
              f"{r['real_time_factor']:.0f}x real time")
        print(f"timings:     {format_timings(r['timings'])}")
    print("OK")
    return 0


def main():
    import sys

    if "--check" in sys.argv:
        sys.exit(check())
    import uvicorn

    log.info("listening on http://%s:%d  (UI: /   API docs: /docs)", settings.HOST, settings.PORT)
    # log_config=None: rknn-toolkit-lite2 rewrites the logging level names on import, which makes uvicorn's
    # dictConfig fail with "Unknown level: 'INFO'"
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_config=None, timeout_keep_alive=30)


if __name__ == "__main__":
    main()
