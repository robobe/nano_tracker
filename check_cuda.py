#!/usr/bin/env python3
"""Verify that NanoTrack's ONNX models execute nodes on CUDA."""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


MODEL_DIR = Path(__file__).parent / "models" / "nanotrackv3"


def check(model: Path, inputs: dict[str, np.ndarray]) -> None:
    options = ort.SessionOptions()
    options.enable_profiling = True
    session = ort.InferenceSession(model, options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"{model.name}: CUDA provider did not load ({session.get_providers()})")
    session.run(None, inputs)
    profile = Path(session.end_profiling())
    try:
        events = json.loads(profile.read_text())
        cuda_nodes = sum(event.get("cat") == "Node" and event.get("args", {}).get("provider") == "CUDAExecutionProvider" for event in events)
    finally:
        profile.unlink(missing_ok=True)
    if not cuda_nodes:
        raise RuntimeError(f"{model.name}: no nodes executed on CUDA")
    print(f"{model.name}: {cuda_nodes} CUDA nodes executed")


def main() -> None:
    print(f"ONNX Runtime {ort.__version__}")
    print(f"available: {ort.get_available_providers()}")
    ort.preload_dlls(directory="")
    search = np.zeros((1, 3, 255, 255), dtype=np.float32)
    check(MODEL_DIR / "nanotrack_backbone.onnx", {"input": search})
    check(
        MODEL_DIR / "nanotrack_head.onnx",
        {
            "input1": np.zeros((1, 96, 8, 8), dtype=np.float32),
            "input2": np.zeros((1, 96, 16, 16), dtype=np.float32),
        },
    )
    print("CUDA ONNX Runtime is working.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        sys.exit(f"CUDA check failed: {error}")
