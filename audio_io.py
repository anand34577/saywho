"""Decode any audio/video file to 16 kHz mono float32.

ffmpeg is used when available: the system one on PATH, else the static binary shipped with the `imageio-ffmpeg`
pip package, so a plain `pip install` is enough on Windows, macOS and Linux (x86_64 and arm64). Without ffmpeg,
WAV / FLAC / OGG / MP3 still work through libsndfile; video and other compressed formats need ffmpeg.
"""
import functools
import shutil
import subprocess

import numpy as np

SAMPLE_RATE = 16000


class AudioDecodeError(ValueError):
    pass


@functools.lru_cache(maxsize=1)
def ffmpeg_path() -> str | None:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - package missing or no binary for this platform
        return None


def load_audio(path: str) -> np.ndarray:
    ffmpeg = ffmpeg_path()
    if ffmpeg:
        return _load_ffmpeg(ffmpeg, path)
    return _load_soundfile(path)


def _load_ffmpeg(ffmpeg: str, path: str) -> np.ndarray:
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path,
           "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        lines = proc.stderr.decode(errors="replace").replace(path, "<input>").strip().splitlines()
        tail = lines[-1] if lines else ""
        raise AudioDecodeError(f"Could not decode audio from this file (ffmpeg): {tail or 'unknown error'}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise AudioDecodeError("The file contains no audio stream.")
    return audio.copy()


def _load_soundfile(path: str) -> np.ndarray:
    import soundfile as sf

    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001
        raise AudioDecodeError(
            f"Could not decode this file without ffmpeg ({e}). Install ffmpeg (or `pip install imageio-ffmpeg`) "
            "to read video and other compressed formats."
        ) from e
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.ascontiguousarray(audio, dtype=np.float32)
