from pathlib import Path

from ..exceptions import EngineUnavailableError, InferenceError, ModelNotFoundError


class ONNXEngine:
    def __init__(self, model_dir: Path, cuda: bool):
        if not model_dir.is_dir():
            raise ModelNotFoundError(f"ONNX model directory not found: {model_dir}")
        backbone = model_dir / "nanotrack_backbone.onnx"
        head = model_dir / "nanotrack_head.onnx"
        if not backbone.is_file() or not head.is_file():
            raise ModelNotFoundError(f"Expected backbone and head ONNX files in {model_dir}")
        try:
            import onnxruntime as ort

            if cuda and hasattr(ort, "preload_dlls"):
                ort.preload_dlls(directory="")
            providers = ort.get_available_providers()
            if cuda and "CUDAExecutionProvider" not in providers:
                raise EngineUnavailableError("CUDA requested but CUDAExecutionProvider is unavailable")
            selected = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
            self.backbone_session = ort.InferenceSession(backbone, providers=selected)
            self.provider = self.backbone_session.get_providers()[0]
            if cuda and self.provider != "CUDAExecutionProvider":
                raise EngineUnavailableError("ONNX Runtime could not initialize CUDA")
            self.head_session = ort.InferenceSession(head, providers=[self.provider])
        except EngineUnavailableError:
            raise
        except Exception as error:
            raise EngineUnavailableError(f"Cannot initialize ONNX Runtime: {error}") from error

    def backbone(self, image):
        try:
            return self.backbone_session.run(None, {"input": image})[0]
        except Exception as error:
            raise InferenceError(f"ONNX backbone inference failed: {error}") from error

    def head(self, template, search):
        try:
            return self.head_session.run(None, {"input1": template, "input2": search})
        except Exception as error:
            raise InferenceError(f"ONNX head inference failed: {error}") from error
