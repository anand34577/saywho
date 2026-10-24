"""Benchmark the NPU with every core mask the SoC supports (run inside the container on the board).

    python bench.py [AUDIO]        default: samples/long.wav (not in the Docker image: pass a file there)

Each mask runs in its own process so the NPU context is released between runs.
Watch the per-core load on the host meanwhile:  sudo watch -n0.5 cat /sys/kernel/debug/rknpu/load
"""
import os
import subprocess
import sys

from diar_engine import RK_PLATFORMS, detect_rk_platform

MASKS = {1: ["auto"], 2: ["0", "1", "0_1"], 3: ["0", "0_1", "0_1_2"]}


def run_one(audio):
    from audio_io import load_audio
    from diar_engine import DiarizationEngine

    data = load_audio(audio)  # any format ffmpeg reads, as 16 kHz mono like the app
    eng = DiarizationEngine("rknn")
    eng.probabilities(data)  # warm-up: the first multi-core runs include one-off NPU setup
    t = min(eng.probabilities(data)[1] for _ in range(3))
    print(f"{eng.accelerator:40s} {t:7.2f} s   {len(data) / 16000 / t:6.1f}x real time  (best of 3)", flush=True)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    audio = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "samples", "long.wav")
    if os.environ.get("_BENCH_CHILD"):
        run_one(audio)
        sys.exit(0)
    if not os.path.isfile(audio):
        sys.exit(f"{audio}: not found (the Docker image has no samples/; pass a file, e.g. one copied in with "
                 "docker cp)")
    platform = detect_rk_platform()
    if platform not in RK_PLATFORMS:
        sys.exit("No Rockchip SoC detected (set RK_PLATFORM=rk3566|rk3576|rk3588)")
    print(f"{platform}: {audio}")
    for mask in MASKS[RK_PLATFORMS[platform][1]]:
        env = dict(os.environ, NPU_CORE_MASK=mask, _BENCH_CHILD="1")
        subprocess.run([sys.executable, __file__, audio], env=env, check=False)
