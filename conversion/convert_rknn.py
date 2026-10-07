"""Convert the exported ONNX step graph to RKNN (FP16) for each Rockchip target. Run on x86 Linux / WSL.

    python conversion/convert_rknn.py [--onnx conversion/nemotron3_diarization.onnx] [--platforms rk3566 rk3576 rk3588]
                                      [--check]

`--check` runs every built model in the rknn-toolkit2 simulator and compares it with onnxruntime.
RK3568 uses the rk3566 model (same NPU).
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(HERE, "nemotron3_diarization.onnx"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "rknn_models"))
    ap.add_argument("--platforms", nargs="+", default=["rk3566", "rk3576", "rk3588"])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    import onnx
    from rknn.api import RKNN

    graph = onnx.load(args.onnx, load_external_data=False).graph
    shapes = [[d.dim_value for d in i.type.tensor_type.shape.dim] for i in graph.input]
    names = [i.name for i in graph.input]
    length, hidden = shapes[0][1], shapes[0][2]
    print("inputs:", dict(zip(names, shapes)))
    os.makedirs(args.out_dir, exist_ok=True)

    if args.check:
        import onnxruntime as ort

        rng = np.random.default_rng(0)
        n = length - 117  # partly padded, like most steps
        x = np.zeros((1, length, hidden), np.float32)
        x[:, :n] = rng.standard_normal((1, n, hidden), dtype=np.float32)
        valid = np.zeros((1, length), np.float32)
        valid[:, :n] = 1
        ref = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"]).run(None, {names[0]: x, names[1]: valid})[0]

    for platform in args.platforms:
        out = os.path.join(args.out_dir, f"nemotron3_diarization_{platform}.rknn")
        rknn = RKNN(verbose=False)
        rknn.config(target_platform=platform, optimization_level=3)
        assert rknn.load_onnx(args.onnx, inputs=names, input_size_list=shapes) == 0, "load_onnx failed"
        assert rknn.build(do_quantization=False) == 0, "build failed"
        assert rknn.export_rknn(out) == 0, "export failed"
        print(f"[{platform}] wrote {out} ({os.path.getsize(out) / 1e6:.0f} MB)")

        if args.check:
            assert rknn.init_runtime() == 0, "simulator init failed"
            got = rknn.inference(inputs=[x, valid])[0]
            m = n * (ref.shape[1] // length)
            d = np.abs(got[:, :m] - ref[:, :m])
            sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
            dp = np.abs(sig(got[:, :m]) - sig(ref[:, :m]))
            print(f"[{platform}] simulator vs onnx: max |dlogit| {d.max():.3f}, max |dprob| {dp.max():.4f}, mean |dprob| {dp.mean():.5f}")
        rknn.release()


if __name__ == "__main__":
    main()
