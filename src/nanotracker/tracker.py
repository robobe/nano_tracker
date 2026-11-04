from enum import Enum
from pathlib import Path

import numpy as np

from .exceptions import EngineUnavailableError, InvalidImageError, TrackerNotInitializedError
from ._version import version
from .types import BoundingBox


class Engine(Enum):
    ONNX = "onnx"
    ONNX_CUDA = "onnx_cuda"
    NCNN = "ncnn"


class NanoTracker:
    """Track one object from OpenCV BGR images and ``(x, y, width, height)`` boxes."""

    def __init__(self, engine: Engine = Engine.ONNX, model_path: str | Path | None = None):
        if not isinstance(engine, Engine):
            raise TypeError("engine must be an Engine value")
        if engine in (Engine.ONNX, Engine.ONNX_CUDA):
            from .engines.onnx import ONNXEngine
            if model_path:
                model_dir = Path(model_path)
            else:
                package_models = Path(__file__).parent / "models" / "nanotrackv3"
                checkout_models = Path(__file__).resolve().parents[2] / "models" / "nanotrackv3"
                model_dir = package_models if package_models.is_dir() else checkout_models
            self.engine = ONNXEngine(model_dir, engine is Engine.ONNX_CUDA)
        else:
            raise EngineUnavailableError("NCNN Python inference is not implemented yet")
        from .core import TrackerCore
        self._core = TrackerCore(self.engine)

    @staticmethod
    def version() -> str:
        """Return the installed NanoTracker distribution version."""
        return version()

    def init(self, image: np.ndarray, bbox: BoundingBox) -> None:
        self._validate_image(image)
        try:
            box = BoundingBox(*(float(value) for value in bbox))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Bounding box must contain four finite numbers") from error
        if not all(np.isfinite(box)) or box.width <= 0 or box.height <= 0:
            raise ValueError("Bounding box must have finite coordinates and positive width and height")
        self._core.initialize(image, box)

    def update(self, image: np.ndarray) -> tuple[BoundingBox, float]:
        self._validate_image(image)
        if not hasattr(self._core, "template"):
            raise TrackerNotInitializedError("Call init(image, bbox) before update(image)")
        box, confidence = self._core.track(image)
        return BoundingBox(*(float(value) for value in box)), confidence

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            raise InvalidImageError("image must be a non-empty HxWx3 NumPy array")
