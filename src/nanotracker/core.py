import cv2
import numpy as np

CONTEXT_AMOUNT = 0.5  # Extra context around the target box.
EXEMPLAR_SIZE = 127  # Template crop size in pixels.
INSTANCE_SIZE = 255  # Search crop size in pixels.
OUTPUT_SIZE = 15  # Matching-head grid width and height.
STRIDE = 16  # Input pixels represented by one output cell.
WINDOW_INFLUENCE = 0.455  # Bias toward locations near the previous target.
PENALTY_K = 0.138  # Penalize abrupt scale or aspect-ratio changes.
LR = 0.348  # Update rate for the tracked box.


def crop(frame: np.ndarray, center: np.ndarray, output_size: int, original_size: int, mean: np.ndarray) -> np.ndarray:
    """Extract a padded square crop and convert it to an NCHW float tensor."""
    half = (original_size + 1) / 2
    x0, y0 = np.floor(center - half + 0.5).astype(int)
    x1, y1 = x0 + original_size - 1, y0 + original_size - 1
    left, top = max(0, -x0), max(0, -y0)
    right, bottom = max(0, x1 - frame.shape[1] + 1), max(0, y1 - frame.shape[0] + 1)
    padded = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=mean.tolist())
    patch = padded[y0 + top : y1 + top + 1, x0 + left : x1 + left + 1]
    if output_size != original_size:
        patch = cv2.resize(patch, (output_size, output_size))
    return patch.transpose(2, 0, 1)[None].astype(np.float32)


class TrackerCore:
    def __init__(self, inference):
        """Load ONNX models and prepare the candidate grid and motion window."""
        self.inference = inference
        coordinates = np.arange(OUTPUT_SIZE) * STRIDE - (OUTPUT_SIZE // 2) * STRIDE
        x, y = np.meshgrid(coordinates, coordinates)
        self.points = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
        window = np.outer(np.hanning(OUTPUT_SIZE), np.hanning(OUTPUT_SIZE))
        self.window = window.ravel()

    def _run_backbone(self, image: np.ndarray) -> np.ndarray:
        """Extract features from a template or search image tensor."""
        return self.inference.backbone(image)

    def _run_head(self, template: np.ndarray, search: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Match template and search features to produce scores and box offsets."""
        return self.inference.head(template, search)

    def initialize(self, frame: np.ndarray, box: tuple[float, float, float, float]) -> None:
        """Store the initial box and extract its template features."""
        x, y, width, height = box
        self.center = np.array([x + (width - 1) / 2, y + (height - 1) / 2])
        self.size = np.array([width, height])
        self.mean = frame.mean(axis=(0, 1))
        context = self.size + CONTEXT_AMOUNT * self.size.sum()
        template_size = round(np.sqrt(context[0] * context[1]))
        template = crop(frame, self.center, EXEMPLAR_SIZE, template_size, self.mean)
        self.template = self._run_backbone(template)

    def reset(self, box: np.ndarray, frame: np.ndarray) -> None:
        """Move the search state to an externally supplied box without replacing its template."""
        x, y, width, height = box
        self.center = np.clip(np.array([x + (width - 1) / 2, y + (height - 1) / 2]), 0, [frame.shape[1], frame.shape[0]])
        self.size = np.clip(np.array([width, height]), 10, [frame.shape[1], frame.shape[0]])

    def track(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Score candidate boxes in a new frame and update the tracked box."""
        context = self.size + CONTEXT_AMOUNT * self.size.sum()
        template_size = np.sqrt(context[0] * context[1])
        scale = EXEMPLAR_SIZE / template_size
        search = crop(frame, self.center, INSTANCE_SIZE, round(template_size * INSTANCE_SIZE / EXEMPLAR_SIZE), self.mean)
        search_features = self._run_backbone(search)
        cls, loc = self._run_head(self.template, search_features)

        logits = cls[0].reshape(2, -1).T
        scores = np.exp(logits[:, 1] - np.logaddexp(logits[:, 0], logits[:, 1]))
        delta = loc[0].reshape(4, -1).copy()
        delta[0] = self.points[:, 0] - delta[0]
        delta[1] = self.points[:, 1] - delta[1]
        delta[2] = self.points[:, 0] + delta[2]
        delta[3] = self.points[:, 1] + delta[3]
        boxes = np.vstack(((delta[0] + delta[2]) / 2, (delta[1] + delta[3]) / 2, delta[2] - delta[0], delta[3] - delta[1]))

        def change(value: np.ndarray) -> np.ndarray:
            """Return a symmetric ratio penalty no smaller than one."""
            return np.maximum(value, 1 / value)

        def padded_size(width: np.ndarray, height: np.ndarray) -> np.ndarray:
            """Calculate NanoTrack's context-padded box size."""
            pad = (width + height) / 2
            return np.sqrt((width + pad) * (height + pad))

        scale_penalty = change(padded_size(boxes[2], boxes[3]) / padded_size(self.size[0] * scale, self.size[1] * scale))
        ratio_penalty = change((self.size[0] / self.size[1]) / (boxes[2] / boxes[3]))
        penalty = np.exp(-(ratio_penalty * scale_penalty - 1) * PENALTY_K)
        best = np.argmax(penalty * scores * (1 - WINDOW_INFLUENCE) + self.window * WINDOW_INFLUENCE)
        box = boxes[:, best] / scale
        learning_rate = penalty[best] * scores[best] * LR
        self.center += box[:2]
        self.size = self.size * (1 - learning_rate) + box[2:] * learning_rate
        self.center = np.clip(self.center, 0, [frame.shape[1], frame.shape[0]])
        self.size = np.clip(self.size, 10, [frame.shape[1], frame.shape[0]])
        return np.r_[self.center - self.size / 2, self.size], float(scores[best])

