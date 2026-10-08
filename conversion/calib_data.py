"""Capture real encoder-step inputs (inputs_embeds + valid) from audio, as the INT8 calibration set for RKNN.

    python conversion/calib_data.py AUDIO [AUDIO ...] [--out conversion/calib]

Runs the full pipeline (speaker cache included) on onnxruntime and saves every padded step the accelerator would
see, then writes `dataset.txt` in the format rknn-toolkit2's `build(dataset=...)` expects (one line per sample, one
.npy per model input).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from diar_engine import DiarizationEngine  # noqa: E402
from validate import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--onnx", default=os.path.join(HERE, "nemotron3_diarization.onnx"))
    ap.add_argument("--out", default=os.path.join(HERE, "calib"))
    args = ap.parse_args()

    eng = DiarizationEngine("onnx", onnx_path=args.onnx)
    step = eng.model.model
    run = step.run
    captured = []

    def recording_run(x, valid):
        captured.append((x.copy(), valid.copy()))
        return run(x, valid)

    step.run = recording_run
    os.makedirs(args.out, exist_ok=True)
    lines = []
    for path in args.audio:
        captured.clear()
        eng.probabilities(load(path))
        stem = os.path.splitext(os.path.basename(path))[0]
        for i, (x, valid) in enumerate(captured):
            xp = os.path.join(args.out, f"{stem}_{i:03d}_x.npy")
            vp = os.path.join(args.out, f"{stem}_{i:03d}_valid.npy")
            np.save(xp, x)
            np.save(vp, valid)
            lines.append(f"{os.path.abspath(xp)} {os.path.abspath(vp)}")
        print(f"{path}: {len(captured)} steps")
    with open(os.path.join(args.out, "dataset.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} samples to {args.out}/dataset.txt")


if __name__ == "__main__":
    main()
