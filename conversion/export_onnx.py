"""Export the encoder + classifier step of Nemotron-3-Diarization to a fixed-shape ONNX graph.

    python conversion/export_onnx.py [--out conversion/nemotron3_diarization.onnx]

Checks, before and after export, that the padded/masked graph matches the original unpadded step.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diar_engine import MODEL_ID, EncoderClassifier, max_frames  # noqa: E402
from transformers import AutoModelForAudioFrameClassification  # noqa: E402


def reference_step(model, embeds):
    """Original transformers path for one unpadded step."""
    out = model.model(inputs_embeds=embeds, position_ids=torch.arange(embeds.shape[1])[None])
    return model.classifier(out.last_hidden_state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "nemotron3_diarization.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = AutoModelForAudioFrameClassification.from_pretrained(MODEL_ID, attn_implementation="eager").eval()
    length = max_frames(model.config)
    hidden = model.config.audio_config.hidden_size
    block = EncoderClassifier(model, length).eval()
    print(f"graph length: {length} encoder frames ({length * 0.08:.2f} s), hidden {hidden}")

    torch.manual_seed(0)
    worst = 0.0
    with torch.no_grad():
        for n in (37, 380, 500, length):
            e = torch.randn(1, n, hidden)
            ref = reference_step(model, e)
            x = torch.zeros(1, length, hidden)
            x[:, :n] = e
            valid = torch.zeros(1, length)
            valid[:, :n] = 1
            got = block(x, valid)[:, : n * 8]
            diff = (got - ref).abs().max().item()
            worst = max(worst, diff)
            print(f"  torch padded vs original, {n:4d} frames: max |dlogit| = {diff:.2e}")
    assert worst < 1e-3, "padded graph diverges from the original"

    x = torch.zeros(1, length, hidden)
    valid = torch.ones(1, length)
    torch.onnx.export(
        block,
        (x, valid),
        args.out,
        input_names=["inputs_embeds", "valid"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB)")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping ONNX check")
        return
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        n = 420
        e = torch.randn(1, n, hidden)
        ref = reference_step(model, e).numpy()
    xn = np.zeros((1, length, hidden), np.float32)
    xn[:, :n] = e.numpy()
    vn = np.zeros((1, length), np.float32)
    vn[:, :n] = 1
    got = sess.run(None, {"inputs_embeds": xn, "valid": vn})[0][:, : n * 8]
    print(f"  onnxruntime vs original, {n} frames: max |dlogit| = {np.abs(got - ref).max():.2e}")


if __name__ == "__main__":
    main()
