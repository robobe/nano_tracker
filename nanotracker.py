#!/usr/bin/env python3
"""Track one selected object in a local video with NanoTrack V3."""

import argparse
import csv
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "nanotrackv3"
BACKBONE = MODEL_DIR / "nanotrack_backbone.onnx"
HEAD = MODEL_DIR / "nanotrack_head.onnx"

CONTEXT_AMOUNT = 0.5
EXEMPLAR_SIZE = 127
INSTANCE_SIZE = 255
OUTPUT_SIZE = 15
STRIDE = 16
WINDOW_INFLUENCE = 0.455
PENALTY_K = 0.138
LR = 0.348


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union for two [x, y, width, height] boxes."""
    left, top = np.maximum(a[:2], b[:2])
    right, bottom = np.minimum(a[:2] + a[2:], b[:2] + b[2:])
    intersection = np.prod(np.maximum(0, [right - left, bottom - top]))
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return float(intersection / union) if union else 0.0


def load_annotations(path: str | None) -> dict[int, np.ndarray]:
    if not path:
        return {}
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"frame", "x", "y", "width", "height"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("annotation CSV must have frame,x,y,width,height headers")
        return {
            int(row["frame"]): np.array(
                [float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"])]
            )
            for row in reader
        }


def crop(frame: np.ndarray, center: np.ndarray, output_size: int, original_size: int, mean: np.ndarray) -> np.ndarray:
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


class NanoTracker:
    def __init__(self, device: str):
        if device != "cpu" and hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        available = ort.get_available_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDA was requested but CUDAExecutionProvider is unavailable")
        providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "auto" and "CUDAExecutionProvider" not in available:
            providers = ["CPUExecutionProvider"]
        self.backbone = ort.InferenceSession(BACKBONE, providers=providers)
        self.provider = self.backbone.get_providers()[0]
        if device == "cuda" and self.provider != "CUDAExecutionProvider":
            raise RuntimeError("CUDA was requested but ONNX Runtime could not initialize it")
        self.head = ort.InferenceSession(HEAD, providers=[self.provider])
        coordinates = np.arange(OUTPUT_SIZE) * STRIDE - (OUTPUT_SIZE // 2) * STRIDE
        x, y = np.meshgrid(coordinates, coordinates)
        self.points = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
        window = np.outer(np.hanning(OUTPUT_SIZE), np.hanning(OUTPUT_SIZE))
        self.window = window.ravel()

    def initialize(self, frame: np.ndarray, box: tuple[float, float, float, float]) -> None:
        x, y, width, height = box
        self.center = np.array([x + (width - 1) / 2, y + (height - 1) / 2])
        self.size = np.array([width, height])
        self.mean = frame.mean(axis=(0, 1))
        context = self.size + CONTEXT_AMOUNT * self.size.sum()
        template_size = round(np.sqrt(context[0] * context[1]))
        template = crop(frame, self.center, EXEMPLAR_SIZE, template_size, self.mean)
        self.template = self.backbone.run(None, {"input": template})[0]

    def track(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        context = self.size + CONTEXT_AMOUNT * self.size.sum()
        template_size = np.sqrt(context[0] * context[1])
        scale = EXEMPLAR_SIZE / template_size
        search = crop(frame, self.center, INSTANCE_SIZE, round(template_size * INSTANCE_SIZE / EXEMPLAR_SIZE), self.mean)
        search_features = self.backbone.run(None, {"input": search})[0]
        cls, loc = self.head.run(None, {"input1": self.template, "input2": search_features})

        logits = cls[0].reshape(2, -1).T
        scores = np.exp(logits[:, 1] - np.logaddexp(logits[:, 0], logits[:, 1]))
        delta = loc[0].reshape(4, -1).copy()
        delta[0] = self.points[:, 0] - delta[0]
        delta[1] = self.points[:, 1] - delta[1]
        delta[2] = self.points[:, 0] + delta[2]
        delta[3] = self.points[:, 1] + delta[3]
        boxes = np.vstack(((delta[0] + delta[2]) / 2, (delta[1] + delta[3]) / 2, delta[2] - delta[0], delta[3] - delta[1]))

        def change(value: np.ndarray) -> np.ndarray:
            return np.maximum(value, 1 / value)

        def padded_size(width: np.ndarray, height: np.ndarray) -> np.ndarray:
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


def draw_box(frame: np.ndarray, box: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    x, y, width, height = np.round(box).astype(int)
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    cv2.putText(frame, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def choose_file(title: str, filetypes: list[tuple[str, str]]) -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


def self_check() -> None:
    assert iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == 1.0
    assert iou(np.array([0, 0, 2, 2]), np.array([3, 3, 2, 2])) == 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not BACKBONE.is_file() or not HEAD.is_file():
        sys.exit("NanoTrack V3 model files are missing from models/nanotrackv3/")

    video_path = choose_file("Choose video", [("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
    if not video_path:
        return
    annotation_path = ""
    root = tk.Tk()
    root.withdraw()
    if messagebox.askyesno("Annotations", "Load an optional annotation CSV?"):
        annotation_path = filedialog.askopenfilename(title="Choose annotation CSV", filetypes=[("CSV files", "*.csv")])
    root.destroy()
    try:
        annotations = load_annotations(annotation_path)
    except (OSError, ValueError) as error:
        sys.exit(f"Cannot load annotations: {error}")

    capture = cv2.VideoCapture(video_path)
    ok, frame = capture.read()
    if not ok:
        sys.exit("Cannot read the selected video")
    window = "NanoTracker"
    box = cv2.selectROI(window, frame, fromCenter=False, showCrosshair=True)
    if not box[2] or not box[3]:
        return
    try:
        tracker = NanoTracker(args.device)
    except Exception as error:
        sys.exit(f"Cannot start ONNX Runtime: {error}")
    tracker.initialize(frame, box)
    print(f"Using {tracker.provider}")

    frame_index = 0
    while True:
        started = time.perf_counter()
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        predicted, confidence = tracker.track(frame)
        fps = 1 / (time.perf_counter() - started)
        draw_box(frame, predicted, (0, 255, 0), f"track {confidence:.2f}")
        truth = annotations.get(frame_index)
        if truth is not None:
            draw_box(frame, truth, (255, 0, 0), f"truth IoU {iou(predicted, truth):.2f}")
        cv2.putText(frame, f"FPS {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
