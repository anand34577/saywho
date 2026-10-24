"""Nemotron-3-Diarization inference engine with pluggable backends.

The offline forward of the model runs chunk by chunk: feature stacking, then for every chunk the speaker cache
(Arrival-Order Speaker Cache + FIFO) is prepended, the transformer encoder + classifier score it, and the cache is
updated. Only the encoder + classifier step is heavy and it is the only part we hand to an accelerator. It is exported
as one fixed-shape graph of MAX_FRAMES encoder frames (cache + fifo + chunk + look-ahead), padded and masked, so the
speaker-cache logic stays the original transformers code on every backend.

Backends:
    auto      best available: rknn -> cuda -> mps -> cpu, each one self-tested before it is used
    rknn      Rockchip NPU via rknn-toolkit-lite2 (RK3566 / RK3568 / RK3576 / RK3588)
    cuda      original model on an NVIDIA GPU
    mps       original model on an Apple Silicon GPU
    cpu       original model on the CPU (any x86_64 / arm64 machine)
    torch     cuda if available, else cpu (reference; kept for compatibility)
    onnx      exported graph on onnxruntime (validation)
    rknn-sim  rknn-toolkit2 simulator on x86 (validation only, slow)
"""
import logging
import os
import threading
import time

import numpy as np
import torch
from torch import nn

import settings
from timing import stage_clock

# transformers >= 5.19 calls torch.accelerator on import, which raises on torch < 2.7 when no GPU is visible
if tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2]) < (2, 7):
    try:
        torch.accelerator.current_accelerator()
    except Exception:
        raise SystemExit(
            f"torch {torch.__version__} is too old for a machine without a usable GPU; install torch >= 2.7 "
            "(see README, 'Install')"
        )

from transformers import AutoModelForAudioFrameClassification, AutoProcessor  # noqa: E402
from transformers.modeling_outputs import BaseModelOutput  # noqa: E402

log = logging.getLogger("diarization")

MODEL_ID = settings.MODEL_ID
SAMPLE_RATE = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
RKNN_DIR = os.environ.get("RKNN_DIR", os.path.join(HERE, "rknn_models"))
MASK_VALUE = -10000.0  # additive attention bias of padded frames; exp() underflows to 0 in fp16 and fp32


def max_frames(config) -> int:
    """Longest encoder input of an offline step: full speaker cache + full fifo + chunk + look-ahead."""
    return (
        config.streaming_config.speaker_cache_length
        + config.fifo_length
        + config.chunk_length
        + config.chunk_right_context
    )


class EncoderClassifier(nn.Module):
    """Encoder + projection + upsampler + classifier of one step at a fixed length, for export.

    Inputs:  inputs_embeds (1, L, 512) float, valid (1, L) float (1 = real frame, 0 = padding)
    Output:  logits (1, L * 8, num_speakers)
    """

    def __init__(self, model, length: int):
        super().__init__()
        tower = model.model.audio_tower
        self.input_layer_norm = tower.input_layer_norm
        self.layers = tower.layers
        self.layer_norm = tower.layer_norm
        self.proj = model.model.proj
        self.upsampler = model.model.upsampler
        self.classifier = model.classifier
        position_ids = torch.arange(length)[None]
        cos, sin = tower.rotary_emb(torch.zeros(1, dtype=torch.float32), position_ids)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, inputs_embeds: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        bias = ((1.0 - valid) * MASK_VALUE)[:, None, None, :]
        hidden = self.input_layer_norm(inputs_embeds)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=bias, position_embeddings=(self.cos, self.sin))
        hidden = self.proj(self.layer_norm(hidden))
        # the upsampler is a k=3 convolution: zero the padding so the last real frame sees what the original
        # (unpadded) graph sees at its border, i.e. the convolution's own zero padding
        hidden = hidden * valid[..., None]
        return self.classifier(self.upsampler(hidden))


class _Tower(nn.Module):
    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder


class AcceleratedStep(nn.Module):
    """Drop-in for `model.model`: pads each step to the fixed graph length and runs it on a backend.

    It returns classifier logits as `last_hidden_state`, so `model.classifier` is replaced by an identity.
    """

    def __init__(self, embedder, run, length: int, hidden_size: int):
        super().__init__()
        self.audio_tower = _Tower(embedder)
        self.run = run
        self.length = length
        self.hidden_size = hidden_size
        self.run_seconds = 0.0  # time spent inside the backend (NPU) since the last reset, for the timing breakdown
        self.run_calls = 0

    def forward(self, inputs_embeds, attention_mask=None, position_ids=None, **kwargs):
        n = inputs_embeds.shape[1]
        if n > self.length:
            raise RuntimeError(f"step of {n} frames exceeds the exported graph length {self.length}")
        x = np.zeros((1, self.length, self.hidden_size), dtype=np.float32)
        x[:, :n] = inputs_embeds.detach().float().cpu().numpy()
        valid = np.zeros((1, self.length), dtype=np.float32)
        valid[:, :n] = 1.0 if attention_mask is None else attention_mask.float().cpu().numpy()
        t0 = time.perf_counter()
        logits = self.run(x, valid)
        self.run_seconds += time.perf_counter() - t0
        self.run_calls += 1
        factor = logits.shape[1] // self.length
        out = torch.from_numpy(np.ascontiguousarray(logits[:, : n * factor], dtype=np.float32))
        return BaseModelOutput(last_hidden_state=out)


# ---------------------------------------------------------------- Rockchip helpers

# platform -> (rknn file suffix, number of NPU cores)
RK_PLATFORMS = {"rk3566": ("rk3566", 1), "rk3568": ("rk3566", 1), "rk3576": ("rk3576", 2), "rk3588": ("rk3588", 3)}


def detect_rk_platform() -> str | None:
    forced = os.environ.get("RK_PLATFORM")
    if forced:
        return forced.lower()
    for path in ("/proc/device-tree/compatible", "/sys/firmware/devicetree/base/compatible"):
        try:
            with open(path, "rb") as f:
                compatible = f.read().decode(errors="ignore").lower()
        except OSError:
            continue
        for name in ("rk3588", "rk3576", "rk3568", "rk3566"):  # rk3588s matches rk3588
            if name in compatible:
                return name
    return None


def rknn_model_path(platform: str) -> str:
    """RKNN_PRECISION: fp16 (default) or int8. The int8 models are experimental: they need about half the NPU memory,
    but on real recordings outside their calibration set they can be badly wrong (e.g. too many speakers)."""
    stem = os.path.join(RKNN_DIR, f"nemotron3_diarization_{RK_PLATFORMS[platform][0]}")
    precision = os.environ.get("RKNN_PRECISION", "fp16").lower()
    return stem + ("_int8.rknn" if precision == "int8" else ".rknn")


def make_rknn_runner(platform: str):
    try:
        from rknnlite.api import RKNNLite
    except (ImportError, OSError) as e:
        raise RuntimeError(
            f"Rockchip NPU runtime not installed ({e}). Docker: set RKNN_ACCEPT_LICENSE=yes; "
            "native: ./install.sh rknn"
        ) from e

    path = rknn_model_path(platform)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing {path}")
    rknn = RKNNLite(verbose=False)
    if rknn.load_rknn(path) != 0:
        raise RuntimeError(f"load_rknn failed for {path}")

    cores = RK_PLATFORMS[platform][1]
    requested = os.environ.get("NPU_CORE_MASK", "all").lower()
    masks = {
        "auto": RKNNLite.NPU_CORE_AUTO,
        "0": RKNNLite.NPU_CORE_0,
        "1": RKNNLite.NPU_CORE_1,
        "2": RKNNLite.NPU_CORE_2,
        "0_1": RKNNLite.NPU_CORE_0_1,
        "0_1_2": RKNNLite.NPU_CORE_0_1_2,
        "all": {1: RKNNLite.NPU_CORE_AUTO, 2: RKNNLite.NPU_CORE_0_1, 3: RKNNLite.NPU_CORE_0_1_2}[cores],
    }
    if cores == 1:  # RK3566/RK3568 have a single core and reject explicit core masks
        requested = "auto"
    if rknn.init_runtime(core_mask=masks[requested]) != 0:
        raise RuntimeError(f"init_runtime failed (core_mask={requested}); is the rknpu driver loaded and "
                           "librknnrt.so in /usr/lib?")

    def run(x, valid):
        return rknn.inference(inputs=[x, valid])[0]

    precision = "int8" if path.endswith("_int8.rknn") else "fp16"
    core_desc = f"{cores} core{'s' if cores > 1 else ''}, mask={requested}, {precision}"
    return run, core_desc


def make_rknn_sim_runner(onnx_path: str, platform: str):
    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    rknn.config(target_platform=RK_PLATFORMS[platform][0])
    if rknn.load_onnx(onnx_path) != 0 or rknn.build(do_quantization=False) != 0 or rknn.init_runtime() != 0:
        raise RuntimeError("rknn simulator setup failed")
    return lambda x, valid: rknn.inference(inputs=[x, valid])[0]


def make_onnx_runner(onnx_path: str):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return lambda x, valid: sess.run(None, {"inputs_embeds": x, "valid": valid})[0]


# ---------------------------------------------------------------- engine


BACKENDS = ("auto", "rknn", "cuda", "mps", "cpu", "torch", "onnx", "rknn-sim")


def _from_pretrained(cls, **kwargs):
    """Local cache first (fast, works offline), then the Hugging Face Hub."""
    try:
        return cls.from_pretrained(MODEL_ID, local_files_only=True, **kwargs)
    except Exception:
        return cls.from_pretrained(MODEL_ID, **kwargs)


def _load_without_encoder():
    """The model minus its transformer encoder layers, for backends that run them in an exported graph.

    Only the feature-stacking embedder and `silence_embeds` are used on the CPU then, so the ~390 MB of fp32 encoder
    weights are never read: what makes the 1 GB RK3566 boards fit.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig

    config = _from_pretrained(AutoConfig)
    try:
        path = hf_hub_download(MODEL_ID, "model.safetensors", local_files_only=True)
    except Exception:
        path = hf_hub_download(MODEL_ID, "model.safetensors")
    with torch.device("meta"):
        model = AutoModelForAudioFrameClassification.from_config(config)
    model.model.audio_tower.layers = nn.ModuleList()
    with safe_open(path, "pt") as f:
        state = {k: f.get_tensor(k) for k in f.keys() if ".audio_tower.layers." not in k}
    model.load_state_dict(state, strict=False, assign=True)
    return model.eval()


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps and mps.is_available())


class DiarizationEngine:
    """Loads the model on the best (or the requested) backend.

    fallback=True: if the requested backend cannot be initialised or fails its self-test, try the next one and
    finally the CPU, recording why in `self.fallbacks`. fallback=False (scripts, benchmarks): fail loudly.
    """

    def __init__(self, backend: str = "auto", onnx_path: str | None = None, fallback: bool = False,
                 cpu_threads: int = 0):
        backend = backend.lower()
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; choose one of {', '.join(BACKENDS)}")
        if cpu_threads > 0:
            torch.set_num_threads(cpu_threads)
        self.processor = _from_pretrained(AutoProcessor)
        self.frame_sec = self.processor.feature_extractor.hop_length / self.processor.feature_extractor.sampling_rate
        self.lock = threading.Lock()
        self.platform = detect_rk_platform()
        self.requested = backend
        self.fallbacks: list[str] = []

        candidates = self._candidates(backend, fallback)
        for i, candidate in enumerate(candidates):
            try:
                self._setup(candidate, onnx_path)
                self._self_test()
                break
            except Exception as e:  # noqa: BLE001 - any failure moves on to the next backend
                reason = f"{candidate}: {type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ''}"
                self.fallbacks.append(reason)
                if i == len(candidates) - 1:
                    raise RuntimeError("no usable backend (" + "; ".join(self.fallbacks) + ")") from e
                log.warning("backend %s unavailable, trying %s (%s)", candidate, candidates[i + 1], reason)
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _candidates(self, backend: str, fallback: bool) -> list[str]:
        if backend == "auto":
            chain = []
            if self.platform in RK_PLATFORMS:
                if os.path.exists(rknn_model_path(self.platform)):
                    chain.append("rknn")
                else:
                    reason = (f"rknn: {self.platform.upper()} detected but {rknn_model_path(self.platform)} is missing; "
                              f"run: python scripts/download_models.py")
                    self.fallbacks.append(reason)
                    log.warning(reason)
            if torch.cuda.is_available():
                chain.append("cuda")
            if _mps_available():
                chain.append("mps")
            return chain + ["cpu"]
        if backend == "torch":
            return ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        return [backend, "cpu"] if fallback and backend != "cpu" else [backend]

    def _setup(self, backend: str, onnx_path: str | None) -> None:
        self.model = None
        accelerated = backend in ("rknn", "rknn-sim", "onnx")
        model = _load_without_encoder() if accelerated else _from_pretrained(AutoModelForAudioFrameClassification).eval()
        self.device = "cpu"
        if backend == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available (no NVIDIA GPU, driver, or CUDA build of PyTorch)")
            device = torch.device(f"cuda:{settings.CUDA_DEVICE}")
            torch.zeros(1, device=device)  # raises if the device or driver is unusable
            model.to(device)
            self.device = str(device)
            self.accelerator = f"{torch.cuda.get_device_name(device)} (CUDA)"
        elif backend == "mps":
            if not _mps_available():
                raise RuntimeError("Apple MPS is not available")
            model.to("mps")
            self.device = "mps"
            self.accelerator = "Apple GPU (MPS)"
        elif backend == "cpu":
            self.accelerator = f"CPU ({torch.get_num_threads()} threads, PyTorch)"
        else:
            length = max_frames(model.config)
            if backend == "rknn":
                if not self.platform:
                    raise RuntimeError("Rockchip SoC not detected; set RK_PLATFORM=rk3566|rk3576|rk3588")
                run, cores = make_rknn_runner(self.platform)
                self.accelerator = f"{self.platform.upper()} NPU ({cores})"
            elif backend == "rknn-sim":
                run = make_rknn_sim_runner(onnx_path, self.platform or "rk3576")
                self.accelerator = "RKNN simulator"
            elif backend == "onnx":
                if not onnx_path or not os.path.exists(onnx_path):
                    raise FileNotFoundError(f"ONNX model not found: {onnx_path} (run conversion/export_onnx.py)")
                run = make_onnx_runner(onnx_path)
                self.accelerator = "ONNX Runtime (CPU)"
            hidden_size = model.config.audio_config.hidden_size
            model.model = AcceleratedStep(model.model.audio_tower.embedder, run, length, hidden_size)
            model.classifier = nn.Identity()  # the accelerated graph already ends with the classifier
            if any(t.is_meta for t in (*model.parameters(), *model.buffers())):
                raise RuntimeError("model weights missing after loading without the encoder")
        self.backend = backend
        self.model = model

    def _self_test(self) -> None:
        """One second of quiet noise through the full pipeline: proves the backend works and warms it up."""
        noise = np.random.default_rng(0).normal(0, 0.01, SAMPLE_RATE).astype(np.float32)
        probs, _ = self.probabilities(noise)
        if probs.ndim != 2 or not np.isfinite(probs).all():
            raise RuntimeError("self-test produced invalid output")

    def info(self) -> dict:
        return {
            "model": MODEL_ID,
            "backend": self.backend,
            "requested_backend": self.requested,
            "device": self.device,
            "accelerator": self.accelerator,
            "fallbacks": self.fallbacks,
            "frame_sec": self.frame_sec,
            "torch": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
        }

    def _forward(self, inputs, device: str):
        logits = self.model(**inputs.to(device)).logits
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        return logits

    def probabilities(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        """Sigmoid speaker probabilities (num_frames, num_speakers) at 10 ms per frame, and inference seconds."""
        probs, timings = self.run(audio)
        return probs, timings["inference"]["wall"]

    @torch.inference_mode()
    def run(self, audio: np.ndarray) -> tuple[np.ndarray, dict]:
        """Probabilities plus a per-stage timing breakdown (see stage_clock)."""
        t = stage_clock()
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        timings = {"features": t.stop()}
        wait = time.perf_counter()
        with self.lock:
            timings["queue_wait"] = round(time.perf_counter() - wait, 3)
            step = self.model.model if isinstance(self.model.model, AcceleratedStep) else None
            if step:
                step.run_seconds, step.run_calls = 0.0, 0
            t = stage_clock()
            try:
                logits = self._forward(inputs, self.device)
            except torch.cuda.OutOfMemoryError:
                # very long file on a small GPU: run this one request on the CPU instead of failing it
                log.warning("GPU out of memory for %.0f s of audio, running this file on the CPU", len(audio) / SAMPLE_RATE)
                torch.cuda.empty_cache()
                self.model.to("cpu")
                try:
                    logits = self._forward(inputs, "cpu")
                finally:
                    self.model.to(self.device)
            timings["inference"] = t.stop()
            if step:
                # time blocked on the accelerator (incl. transfers); the rest of `inference` is CPU-side work
                # (embedding, speaker cache, padding) around it
                timings["accelerator"] = {"wall": round(step.run_seconds, 3), "calls": step.run_calls}
                timings["cpu_side"] = {"wall": round(max(0.0, timings["inference"]["wall"] - step.run_seconds), 3)}
        return logits[0].sigmoid().float().cpu().numpy(), timings
