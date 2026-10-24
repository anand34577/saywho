"""Build an RKNN model in the simulator and score it end to end against the saved reference probabilities.

    python conversion/quant_eval.py [--quant int8|fp16] [--platform rk3566] [--export path.rknn]

Every recording in samples/ with a matching <name>_ref.npy is run through the full pipeline (speaker cache
included). Make a reference on a GPU or CPU first:
    python conversion/validate.py samples/x.wav --backend onnx --save-reference samples/x_ref.npy
The simulator cannot load an exported .rknn, so the model is built here exactly as convert_rknn.py builds it.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from convert_rknn import FP16_TENSORS, build_int8  # noqa: E402
from diar_engine import DiarizationEngine  # noqa: E402
from validate import load, segments  # noqa: E402

NAMES, SHAPES = ["inputs_embeds", "valid"], [[1, 684, 512], [1, 684]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(HERE, "nemotron3_diarization.onnx"))
    ap.add_argument("--platform", default="rk3566")
    ap.add_argument("--quant", choices=["fp16", "int8"], default="int8")
    ap.add_argument("--dataset", default=os.path.join(HERE, "calib", "dataset.txt"))
    ap.add_argument("--export", help="also write the built model here")
    ap.add_argument("--fp16", help="regex of tensors kept fp16 (default: convert_rknn.FP16_TENSORS)")
    args = ap.parse_args()

    from rknn.api import RKNN

    if args.quant == "int8":
        rknn = build_int8(RKNN, args.platform, os.path.abspath(args.onnx), NAMES, SHAPES, os.path.abspath(args.dataset),
                          re.compile(args.fp16) if args.fp16 else FP16_TENSORS)
    else:
        rknn = RKNN(verbose=False)
        rknn.config(target_platform=args.platform, optimization_level=3)
        assert rknn.load_onnx(args.onnx, inputs=NAMES, input_size_list=SHAPES) == 0
        assert rknn.build(do_quantization=False) == 0
    if args.export:
        assert rknn.export_rknn(args.export) == 0
        print(f"wrote {args.export} ({os.path.getsize(args.export) / 1e6:.0f} MB)")
    assert rknn.init_runtime() == 0

    eng = DiarizationEngine("onnx", onnx_path=args.onnx)
    eng.model.model.run = lambda x, valid: rknn.inference(inputs=[x, valid])[0]
    for ref_path in sorted(glob.glob(os.path.join(ROOT, "samples", "*_ref.npy"))):
        stem = ref_path[: -len("_ref.npy")]
        wav = next((stem + ext for ext in (".wav", ".flac", ".mp3", ".m4a") if os.path.exists(stem + ext)), stem + ".wav")
        ref = np.load(ref_path)
        got, _ = eng.probabilities(load(wav))
        diff = np.abs(got - ref)
        flips = ((got >= 0.5) != (ref >= 0.5)).any(axis=1).mean()
        sr, sg = segments(ref, eng.frame_sec), segments(got, eng.frame_sec)
        close = sum(any(c == d and abs(s - t) <= 0.1 and abs(e - u) <= 0.1 for t, u, d in sg) for s, e, c in sr)
        print(f"[{args.platform} {args.quant}] {os.path.basename(wav)}: max |dprob| {diff.max():.4f}, "
              f"mean {diff.mean():.5f}, frames differing {flips * 100:.3f}%, segments ref {len(sr)} / got {len(sg)} / "
              f"identical {len(set(sr) & set(sg))}, within 0.1 s {close}, speakers ref {len({s[2] for s in sr})} got {len({s[2] for s in sg})}")


if __name__ == "__main__":
    main()
