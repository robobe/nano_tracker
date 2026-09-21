#!/usr/bin/env python3
"""Convert NanoTrack V3 ONNX models into static FP16 RK3566 RKNN models.

Flow: ONNX backbone/head -> three fixed-shape RKNN models -> optional numerical check.
Run this on the x86 conversion host, not on the Radxa board.
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rknn.api import RKNN


ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "models" / "nanotrackv3"
OUTPUT_DIR = ROOT / "models" / "nanotrackv3_rknn_rk3566"

MODELS = (
    ("backbone_template.rknn", ONNX_DIR / "nanotrack_backbone.onnx", ["input"], [[1, 3, 127, 127]]),
    ("backbone_search.rknn", ONNX_DIR / "nanotrack_backbone.onnx", ["input"], [[1, 3, 255, 255]]),
    ("head.rknn", ONNX_DIR / "nanotrack_head.onnx", ["input1", "input2"], [[1, 96, 8, 8], [1, 96, 16, 16]]),
)


def require_ok(result: int, action: str) -> None:
    if result != 0:
        raise RuntimeError(f"RKNN failed to {action} (code {result})")


def convert(name: str, onnx_path: Path, inputs: list[str], shapes: list[list[int]], output_dir: Path, validate: bool) -> None:
    output_path = output_dir / name
    rknn = RKNN(verbose=False)
    try:
        require_ok(rknn.config(target_platform="rk3566"), "configure RK3566")
        require_ok(rknn.load_onnx(model=str(onnx_path), inputs=inputs, input_size_list=shapes), f"load {onnx_path.name}")
        require_ok(rknn.build(do_quantization=False), f"build {name}")
        require_ok(rknn.export_rknn(str(output_path)), f"export {name}")
        if validate:
            samples = [np.random.default_rng(0).random(shape, dtype=np.float32) * 255 for shape in shapes]
            reference = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(None, dict(zip(inputs, samples)))
            require_ok(rknn.init_runtime(), f"start RKNN simulator for {name}")
            result = rknn.inference(inputs=samples, data_format="nchw")
            for index, (expected, actual) in enumerate(zip(reference, result, strict=True)):
                if expected.shape != actual.shape or not np.isfinite(actual).all():
                    raise RuntimeError(f"{name} output {index} has an invalid shape or value")
                print(f"{name} output {index}: max abs difference {np.max(np.abs(expected - actual)):.5f}")
    finally:
        rknn.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-validate", action="store_true", help="skip the host RKNN/ONNX comparison")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, onnx_path, inputs, shapes in MODELS:
        convert(name, onnx_path, inputs, shapes, args.output_dir, not args.no_validate)


if __name__ == "__main__":
    main()
