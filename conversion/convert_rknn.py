"""Convert the exported ONNX step graph to RKNN for each Rockchip target. Run on x86 Linux / WSL.

    python conversion/convert_rknn.py [--onnx conversion/nemotron3_diarization.onnx] [--platforms rk3566 rk3576 rk3588]
                                      [--quant fp16|int8] [--dataset conversion/calib/dataset.txt] [--check]

fp16  (default) nemotron3_diarization_<platform>.rknn, ~230-265 MB
int8  nemotron3_diarization_<platform>_int8.rknn, ~125 MB: hybrid quantization, weights and activations int8 except
      the attention score path (QK^T, mask, softmax, attention x V), which stays fp16. Those tensors have no weights,
      but plain int8 breaks them: the -10000 padding mask stretches the score range so far that real scores collapse
      to one level, and most of the 684 attention weights are smaller than one int8 step of [0, 1].
      Needs a calibration set: python conversion/calib_data.py RECORDING [RECORDING ...]  (a few minutes of real speech)

`--check` runs every built model in the rknn-toolkit2 simulator and compares it with onnxruntime.
RK3568 uses the rk3566 model (same NPU).
"""
import argparse
import os
import re
import shutil
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FP16_TENSORS = re.compile(r"/self_attn/(Mul_4|Add_2|Softmax|MatMul_1)_output_0")


def build_int8(RKNN, platform, onnx_path, names, shapes, dataset, fp16_tensors=FP16_TENSORS):
    """Hybrid quantization: step 1 writes the per-tensor config, which we edit to keep the attention path fp16."""
    work = tempfile.mkdtemp(prefix="rknn_hybrid_")
    cwd = os.getcwd()
    os.chdir(work)  # step 1 writes its files into the working directory
    try:
        rknn = RKNN(verbose=False)
        rknn.config(target_platform=platform, optimization_level=3, quantized_dtype="w8a8")
        assert rknn.load_onnx(onnx_path, inputs=names, input_size_list=shapes) == 0, "load_onnx failed"
        assert rknn.hybrid_quantization_step1(dataset=dataset, proposal=False) == 0, "hybrid step 1 failed"
        rknn.release()
        stem = os.path.splitext(os.path.basename(onnx_path))[0]
        cfg = os.path.join(work, f"{stem}.quantization.cfg")
        with open(cfg) as f:
            text = f.read()
        layers = sorted({m.group(1) for m in re.finditer(r"^    (\S+):$", text, re.M) if fp16_tensors.search(m.group(1))})
        assert layers, "no attention tensors found in the quantization config"
        custom = "custom_quantize_layers:\n" + "".join(f"    {layer}: float16\n" for layer in layers)
        with open(cfg, "w") as f:
            f.write(text.replace("custom_quantize_layers: {}\n", custom, 1))
        print(f"[{platform}] keeping {len(layers)} attention tensors in fp16")

        rknn = RKNN(verbose=False)
        assert rknn.hybrid_quantization_step2(
            model_input=os.path.join(work, f"{stem}.model"),
            data_input=os.path.join(work, f"{stem}.data"),
            model_quantization_cfg=cfg,
        ) == 0, "hybrid step 2 failed"
        return rknn
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(HERE, "nemotron3_diarization.onnx"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "rknn_models"))
    ap.add_argument("--platforms", nargs="+", default=["rk3566", "rk3576", "rk3588"])
    ap.add_argument("--quant", choices=["fp16", "int8"], default="fp16")
    ap.add_argument("--dataset", default=os.path.join(HERE, "calib", "dataset.txt"))
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
    if args.quant == "int8" and not os.path.exists(args.dataset):
        raise SystemExit(f"calibration set {args.dataset} missing; run conversion/calib_data.py first")

    if args.check:
        import onnxruntime as ort

        if args.quant == "int8":  # quantization is only meaningful on real inputs
            x, valid = (np.load(p) for p in open(args.dataset).readline().split())
            n = int(valid.sum())
        else:
            rng = np.random.default_rng(0)
            n = length - 117  # partly padded, like most steps
            x = np.zeros((1, length, hidden), np.float32)
            x[:, :n] = rng.standard_normal((1, n, hidden), dtype=np.float32)
            valid = np.zeros((1, length), np.float32)
            valid[:, :n] = 1
        ref = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"]).run(None, {names[0]: x, names[1]: valid})[0]

    for platform in args.platforms:
        suffix = "_int8" if args.quant == "int8" else ""
        out = os.path.join(args.out_dir, f"nemotron3_diarization_{platform}{suffix}.rknn")
        if args.quant == "int8":
            rknn = build_int8(RKNN, platform, args.onnx, names, shapes, args.dataset)
        else:
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
