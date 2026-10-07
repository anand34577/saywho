"""Compare a backend against the reference PyTorch model on real audio, end to end (speaker cache included).

    python conversion/validate.py AUDIO --backend onnx|rknn-sim|rknn [--onnx path] [--platform rk3576]
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diar_engine import SAMPLE_RATE, DiarizationEngine  # noqa: E402


def load(path):
    info = sf.info(path) if path.lower().endswith(".wav") else None
    if info and info.samplerate == SAMPLE_RATE and info.channels == 1:
        return sf.read(path, dtype="float32")[0]
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "a.wav")
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE), wav], check=True)
        audio, _ = sf.read(wav, dtype="float32")
    return audio


def segments(probs, frame_sec, thr=0.5):
    out = []
    for c in range(probs.shape[1]):
        a = np.concatenate([[0], (probs[:, c] >= thr).astype(int), [0]])
        d = np.diff(a)
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if (e - s) * frame_sec >= 0.1:
                out.append((round(s * frame_sec, 2), round(e * frame_sec, 2), c))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--backend", required=True)
    ap.add_argument("--onnx", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "nemotron3_diarization.onnx"))
    ap.add_argument("--platform")
    ap.add_argument("--reference", help="saved reference probabilities (.npy) instead of running PyTorch")
    ap.add_argument("--save-reference", help="write the PyTorch probabilities to this .npy")
    args = ap.parse_args()
    if args.platform:
        os.environ["RK_PLATFORM"] = args.platform

    audio = load(args.audio)
    print(f"{args.audio}: {len(audio) / SAMPLE_RATE:.1f} s")

    if args.reference:
        ref = np.load(args.reference)
    else:
        ref_engine = DiarizationEngine("torch")
        ref, t = ref_engine.probabilities(audio)
        print(f"reference torch/{ref_engine.device}: {t:.2f} s")
        if args.save_reference:
            np.save(args.save_reference, ref)
        del ref_engine

    eng = DiarizationEngine(args.backend, onnx_path=args.onnx)
    got, t = eng.probabilities(audio)
    print(f"{args.backend} ({eng.accelerator}): {t:.2f} s, {len(audio) / SAMPLE_RATE / t:.1f}x real time")

    diff = np.abs(got - ref)
    flips = ((got >= 0.5) != (ref >= 0.5)).any(axis=1).mean()
    print(f"max |dprob| = {diff.max():.4f}, mean |dprob| = {diff.mean():.5f}, frames with a different decision = {flips * 100:.3f}%")
    sr, sg = segments(ref, eng.frame_sec), segments(got, eng.frame_sec)
    print(f"segments: reference {len(sr)}, {args.backend} {len(sg)}, identical lists: {sr == sg}")
    print(f"speakers: reference {sorted({s[2] for s in sr})}, {args.backend} {sorted({s[2] for s in sg})}")


if __name__ == "__main__":
    main()
