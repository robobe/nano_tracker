#!/usr/bin/env python3
"""Track one selected object in a local video with NanoTrack V3.

The user selects an initial box, the backbone extracts template and search features,
and the head ranks candidate boxes on each following frame. The best candidate updates
the tracked box, which OpenCV displays alongside optional ground-truth annotations.
"""

import argparse
import base64
import csv
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "nanotrackv3"
BACKBONE = MODEL_DIR / "nanotrack_backbone.onnx"
HEAD = MODEL_DIR / "nanotrack_head.onnx"

CONTEXT_AMOUNT = 0.5  # Extra context around the target box.
EXEMPLAR_SIZE = 127  # Template crop size in pixels.
INSTANCE_SIZE = 255  # Search crop size in pixels.
OUTPUT_SIZE = 15  # Matching-head grid width and height.
STRIDE = 16  # Input pixels represented by one output cell.
WINDOW_INFLUENCE = 0.455  # Bias toward locations near the previous target.
PENALTY_K = 0.138  # Penalize abrupt scale or aspect-ratio changes.
LR = 0.348  # Update rate for the tracked box.


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union for two [x, y, width, height] boxes."""
    left, top = np.maximum(a[:2], b[:2])
    right, bottom = np.minimum(a[:2] + a[2:], b[:2] + b[2:])
    intersection = np.prod(np.maximum(0, [right - left, bottom - top]))
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return float(intersection / union) if union else 0.0


def load_annotations(path: str | None) -> dict[int, np.ndarray]:
    """Load frame-indexed ground-truth boxes from an optional CSV file."""
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


class KalmanBoxEstimator:
    """Estimate a box using constant centre/size velocity."""

    def __init__(self) -> None:
        self.filter = cv2.KalmanFilter(8, 4)
        self.filter.transitionMatrix = np.eye(8, dtype=np.float32)
        self.filter.transitionMatrix[:4, 4:] = np.eye(4, dtype=np.float32)
        self.filter.measurementMatrix = np.hstack((np.eye(4), np.zeros((4, 4)))).astype(np.float32)
        self.filter.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        self.filter.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.filter.errorCovPost = np.eye(8, dtype=np.float32)

    @staticmethod
    def _state(box: np.ndarray | tuple[float, float, float, float]) -> np.ndarray:
        x, y, width, height = box
        return np.array([x + width / 2, y + height / 2, width, height], dtype=np.float32)

    @staticmethod
    def _box(state: np.ndarray) -> np.ndarray:
        center_x, center_y, width, height = state[:4].ravel()
        width, height = max(1.0, float(width)), max(1.0, float(height))
        return np.array([center_x - width / 2, center_y - height / 2, width, height], dtype=np.float32)

    def initialize(self, box: np.ndarray | tuple[float, float, float, float]) -> None:
        state = self._state(box)
        self.filter.statePost = np.r_[state, np.zeros(4, dtype=np.float32)].reshape(8, 1)

    def predict(self) -> np.ndarray:
        return self._box(self.filter.predict())

    def correct(self, measurement: np.ndarray) -> np.ndarray:
        return self._box(self.filter.correct(self._state(measurement).reshape(4, 1)))


def should_correct(estimate: np.ndarray, measurement: np.ndarray, enabled: bool, threshold: float) -> bool:
    """Return whether a low-overlap measurement should be rejected."""
    return enabled and iou(estimate, measurement) < threshold


def matching_annotation(video_path: str | Path) -> Path | None:
    """Return the sibling CSV sharing the selected video's basename, when present."""
    candidate = Path(video_path).with_suffix(".csv")
    return candidate if candidate.is_file() else None


class NanoTracker:
    def __init__(self, device: str):
        """Load ONNX models and prepare the candidate grid and motion window."""
        import onnxruntime as ort

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

    def _run_backbone(self, image: np.ndarray) -> np.ndarray:
        """Extract features from a template or search image tensor."""
        return self.backbone.run(None, {"input": image})[0]

    def _run_head(self, template: np.ndarray, search: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Match template and search features to produce scores and box offsets."""
        return self.head.run(None, {"input1": template, "input2": search})

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


def draw_box(frame: np.ndarray, box: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    """Draw a labeled bounding box on an OpenCV frame."""
    x, y, width, height = np.round(box).astype(int)
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    cv2.putText(frame, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def self_check() -> None:
    """Check the bounding-box overlap helper on known cases."""
    assert iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == 1.0
    assert iou(np.array([0, 0, 2, 2]), np.array([3, 3, 2, 2])) == 0.0


def parse_roi(value: str) -> tuple[float, float, float, float]:
    """Parse and validate an ``x,y,width,height`` command-line ROI."""
    try:
        box = tuple(float(number) for number in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height") from error
    if len(box) != 4 or box[2] <= 0 or box[3] <= 0:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height with positive width and height")
    return box


def warm_tracker(device: str) -> NanoTracker:
    """Load and exercise both models before the user begins drawing an ROI."""
    tracker = NanoTracker(device)
    frame = np.zeros((255, 255, 3), dtype=np.uint8)
    tracker.initialize(frame, (80, 80, 64, 64))
    tracker.track(frame)
    return tracker


class TrackerApp:
    """One-window GUI for video selection, ROI selection, and live tracking."""

    PREVIEW_WIDTH = 960
    PREVIEW_HEIGHT = 540

    def __init__(self, root, device: str, video_path: str | None = None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root, self.tk, self.ttk, self.device = root, tk, ttk, device
        self.video_path = tk.StringVar(value=video_path or "")
        self.annotation_path = tk.StringVar()
        self.show_tracker = tk.BooleanVar(value=True)
        self.show_annotations = tk.BooleanVar(value=True)
        self.show_estimate = tk.BooleanVar(value=True)
        self.correct_tracker = tk.BooleanVar()
        self.threshold = tk.StringVar(value="0.20")
        self.status = tk.StringVar(value="Warming ONNX Runtime…")
        self.capture = None
        self.first_frame: np.ndarray | None = None
        self.annotations: dict[int, np.ndarray] = {}
        self.roi: tuple[float, float, float, float] | None = None
        self.drag_start: tuple[float, float] | None = None
        self.roi_shape = None
        self.frame_shape = None
        self.preview_scale = 1.0
        self.preview_offset = (0, 0)
        self.photo = None
        self.tracker: NanoTracker | None = None
        self.estimator: KalmanBoxEstimator | None = None
        self.running = False
        self.paused = False
        self.frame_index = 0
        self.warm_results: queue.Queue[tuple[str, object]] = queue.Queue()

        root.title("NanoTracker")
        root.columnconfigure(1, weight=1)
        root.rowconfigure(4, weight=1)
        self._path_row("Video", self.video_path, self.choose_video, 0, "Choose video")
        self.ttk.Label(root, text="Annotations").grid(row=1, column=0, padx=8, pady=4, sticky="w")
        self.annotation_entry = ttk.Entry(root, textvariable=self.annotation_path)
        self.annotation_entry.grid(row=1, column=1, padx=8, pady=4, sticky="ew")
        self.annotation_button = ttk.Button(root, text="Choose CSV", command=self.choose_annotations)
        self.annotation_button.grid(row=1, column=2, padx=8, pady=4)
        controls = ttk.Frame(root)
        controls.grid(row=2, column=0, columnspan=3, padx=8, pady=4, sticky="ew")
        ttk.Checkbutton(controls, text="Show tracker", variable=self.show_tracker).pack(side="left")
        ttk.Checkbutton(controls, text="Show annotation", variable=self.show_annotations).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(controls, text="Show estimate", variable=self.show_estimate).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(controls, text="Correct tracker on disagreement", variable=self.correct_tracker).pack(side="left", padx=(12, 4))
        ttk.Label(controls, text="IoU threshold").pack(side="left", padx=(8, 4))
        ttk.Spinbox(controls, from_=0.01, to=0.99, increment=0.01, textvariable=self.threshold, width=5).pack(side="left")
        buttons = ttk.Frame(root)
        buttons.grid(row=3, column=0, columnspan=3, padx=8, pady=4, sticky="e")
        self.start_button = ttk.Button(buttons, text="Start", command=self.start, state="disabled")
        self.start_button.pack(side="left", padx=4)
        self.pause_button = ttk.Button(buttons, text="Pause", command=self.pause, state="disabled")
        self.pause_button.pack(side="left", padx=4)
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=4)
        self.canvas = tk.Canvas(root, width=self.PREVIEW_WIDTH, height=self.PREVIEW_HEIGHT, background="black", highlightthickness=0)
        self.canvas.grid(row=4, column=0, columnspan=3, padx=8, pady=4, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self.begin_roi)
        self.canvas.bind("<B1-Motion>", self.drag_roi)
        self.canvas.bind("<ButtonRelease-1>", self.finish_roi)
        ttk.Label(root, textvariable=self.status).grid(row=5, column=0, columnspan=3, padx=8, pady=(0, 8), sticky="w")

        threading.Thread(target=self._warm, daemon=True).start()
        root.after(50, self.poll_warmup)
        if video_path:
            self.load_video()

    def _path_row(self, label, variable, command, row, button_label) -> None:
        self.ttk.Label(self.root, text=label).grid(row=row, column=0, padx=8, pady=4, sticky="w")
        self.ttk.Entry(self.root, textvariable=variable).grid(row=row, column=1, padx=8, pady=4, sticky="ew")
        self.ttk.Button(self.root, text=button_label, command=command).grid(row=row, column=2, padx=8, pady=4)

    def _warm(self) -> None:
        try:
            self.warm_results.put(("ready", warm_tracker(self.device)))
        except Exception as error:
            self.warm_results.put(("error", error))

    def poll_warmup(self) -> None:
        try:
            kind, value = self.warm_results.get_nowait()
        except queue.Empty:
            self.root.after(50, self.poll_warmup)
            return
        if kind == "error":
            self.status.set(f"Cannot start ONNX Runtime: {value}")
            return
        self.tracker = value
        self.status.set(f"Ready: {self.tracker.provider}. Choose a video and draw its ROI.")
        if self.first_frame is not None:
            self.start_button.configure(state="normal")

    def choose_video(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(title="Choose video", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if path:
            self.video_path.set(path)
            self.load_video()

    def choose_annotations(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(title="Choose annotation CSV", filetypes=[("CSV files", "*.csv")])
        if path:
            self.annotation_path.set(path)

    def load_video(self) -> None:
        from tkinter import messagebox

        self.stop()
        if self.capture:
            self.capture.release()
        self.capture = cv2.VideoCapture(self.video_path.get())
        ok, self.first_frame = self.capture.read()
        if not ok:
            self.first_frame = None
            messagebox.showerror("Cannot read video", "Cannot read the selected video.")
            return
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 1)
        self.roi = None
        self.frame_index = 0
        annotation = matching_annotation(self.video_path.get())
        self.annotation_path.set(str(annotation) if annotation else "")
        self.show_frame(self.first_frame)
        if self.tracker:
            self.start_button.configure(state="normal")
        self.status.set("Draw a box around the target, then press Start.")

    def begin_roi(self, event) -> None:
        if self.first_frame is None or self.running:
            return
        self.drag_start = (event.x, event.y)
        if self.roi_shape:
            self.canvas.delete(self.roi_shape)
        self.roi_shape = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="white", width=2)

    def drag_roi(self, event) -> None:
        if self.drag_start and self.roi_shape:
            self.canvas.coords(self.roi_shape, *self.drag_start, event.x, event.y)

    def finish_roi(self, event) -> None:
        if not self.drag_start or self.first_frame is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        self.drag_start = None
        left, top = min(x0, x1), min(y0, y1)
        width, height = abs(x1 - x0), abs(y1 - y0)
        if width < 2 or height < 2:
            self.status.set("Draw a larger target box.")
            return
        offset_x, offset_y = self.preview_offset
        self.roi = ((left - offset_x) / self.preview_scale, (top - offset_y) / self.preview_scale, width / self.preview_scale, height / self.preview_scale)
        self.status.set("ROI selected. Press Start.")

    def start(self) -> None:
        from tkinter import messagebox

        if not self.tracker:
            self.status.set("Still warming ONNX Runtime…")
            return
        if self.first_frame is None or not self.capture:
            messagebox.showerror("Choose video", "Choose a readable video first.")
            return
        if not self.roi:
            messagebox.showerror("Draw ROI", "Draw a box around the target first.")
            return
        try:
            threshold = float(self.threshold.get())
            if not 0 < threshold < 1:
                raise ValueError
            self.annotations = load_annotations(self.annotation_path.get()) if self.annotation_path.get() else {}
        except (OSError, ValueError):
            messagebox.showerror("Annotations or threshold", "Use a valid annotation CSV and an IoU threshold between 0 and 1.")
            return
        self.tracker.initialize(self.first_frame, self.roi)
        self.estimator = KalmanBoxEstimator()
        self.estimator.initialize(self.roi)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 1)
        self.frame_index = 0
        self.running, self.paused = True, False
        if self.roi_shape:
            self.canvas.delete(self.roi_shape)
            self.roi_shape = None
        self.start_button.configure(state="disabled")
        self.pause_button.configure(state="normal", text="Pause")
        self.stop_button.configure(state="normal")
        self.status.set(f"Tracking with {self.tracker.provider}.")
        self.root.after(1, self.track_next)

    def pause(self) -> None:
        if not self.running:
            return
        self.paused = not self.paused
        self.pause_button.configure(text="Resume" if self.paused else "Pause")
        self.status.set("Paused." if self.paused else "Tracking…")
        if not self.paused:
            self.root.after(1, self.track_next)

    def stop(self) -> None:
        was_running = self.running
        self.running, self.paused = False, False
        self.pause_button.configure(state="disabled", text="Pause")
        self.stop_button.configure(state="disabled")
        if self.tracker and self.first_frame is not None:
            self.start_button.configure(state="normal")
        if was_running:
            self.show_frame(self.first_frame)
            self.status.set("Stopped. Draw a new ROI or press Start to restart.")

    def track_next(self) -> None:
        if not self.running or self.paused or not self.capture or not self.tracker or not self.estimator:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.stop()
            self.status.set("Video complete.")
            return
        started = time.perf_counter()
        estimate = self.estimator.predict()
        measurement, confidence = self.tracker.track(frame)
        corrected = should_correct(estimate, measurement, self.correct_tracker.get(), float(self.threshold.get()))
        if corrected:
            self.tracker.reset(estimate, frame)
        else:
            self.estimator.correct(measurement)
        self.frame_index += 1
        display = frame.copy()
        if self.show_tracker.get():
            draw_box(display, measurement, (0, 255, 0), f"track {confidence:.2f}")
        if self.show_estimate.get():
            draw_box(display, estimate, (0, 255, 255), "estimate")
        truth = self.annotations.get(self.frame_index)
        if truth is not None and self.show_annotations.get():
            draw_box(display, truth, (255, 0, 0), f"truth IoU {iou(measurement, truth):.2f}")
        elapsed = time.perf_counter() - started
        suffix = " corrected" if corrected else ""
        cv2.putText(display, f"FPS {1 / elapsed:.1f}{suffix}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        self.show_frame(display)
        self.root.after(1, self.track_next)

    def show_frame(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        self.preview_scale = min(self.PREVIEW_WIDTH / width, self.PREVIEW_HEIGHT / height)
        preview_size = (round(width * self.preview_scale), round(height * self.preview_scale))
        preview = cv2.resize(frame, preview_size)
        offset_x = (self.PREVIEW_WIDTH - preview_size[0]) // 2
        offset_y = (self.PREVIEW_HEIGHT - preview_size[1]) // 2
        self.preview_offset = (offset_x, offset_y)
        canvas_frame = np.zeros((self.PREVIEW_HEIGHT, self.PREVIEW_WIDTH, 3), dtype=np.uint8)
        canvas_frame[offset_y : offset_y + preview_size[1], offset_x : offset_x + preview_size[0]] = preview
        ok, encoded = cv2.imencode(".png", canvas_frame)
        if not ok:
            return
        self.photo = self.tk.PhotoImage(data=base64.b64encode(encoded.tobytes()), format="png")
        if self.frame_shape:
            self.canvas.itemconfigure(self.frame_shape, image=self.photo)
        else:
            self.frame_shape = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.tag_lower(self.frame_shape)
        if self.roi and not self.running:
            x, y, width, height = self.roi
            self.roi_shape = self.canvas.create_rectangle(x * self.preview_scale + offset_x, y * self.preview_scale + offset_y, (x + width) * self.preview_scale + offset_x, (y + height) * self.preview_scale + offset_y, outline="white", width=2)


def main() -> None:
    """Run the one-screen GUI or preserve the existing headless mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--input", help="local video path; omitting it opens the file chooser")
    parser.add_argument("--roi", type=parse_roi, help="initial box: x,y,width,height")
    parser.add_argument("--no-display", action="store_true", help="run without OpenCV windows")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not BACKBONE.is_file() or not HEAD.is_file():
        sys.exit("NanoTrack V3 model files are missing from models/nanotrackv3/")
    if args.no_display and (not args.input or not args.roi):
        sys.exit("--no-display requires --input and --roi")

    if not args.no_display:
        import tkinter as tk

        root = tk.Tk()
        TrackerApp(root, args.device, args.input)
        root.mainloop()
        return

    capture = cv2.VideoCapture(args.input)
    ok, frame = capture.read()
    if not ok:
        sys.exit("Cannot read the selected video")
    try:
        tracker = NanoTracker(args.device)
    except Exception as error:
        sys.exit(f"Cannot start ONNX Runtime: {error}")
    tracker.initialize(frame, args.roi)
    print(f"Using {tracker.provider}")

    frame_index = 0
    total_seconds = 0.0
    while True:
        started = time.perf_counter()
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        predicted, confidence = tracker.track(frame)
        elapsed = time.perf_counter() - started
        total_seconds += elapsed
        fps = 1 / elapsed
    capture.release()
    if frame_index:
        print(f"Processed {frame_index} frames at {frame_index / total_seconds:.1f} FPS")


if __name__ == "__main__":
    main()
