#!/usr/bin/env python3
"""Convert a numbered JPEG sequence to MP4, with an optional NanoTracker CSV."""

from __future__ import annotations

import csv
import json
import os
import queue
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import yaml


ROOT = Path(__file__).resolve().parents[1]
PRESETS = Path(__file__).with_name("video_presets.yaml")
DEFAULT_SOURCE_DIR = Path.home() / "datasets"
IMAGE_NAME = re.compile(r"^(\d+)\.jpg$", re.IGNORECASE)


@dataclass(frozen=True)
class ImageSequence:
    directory: Path
    pattern: str
    start_number: int
    count: int
    width: int
    height: int


def load_presets(path: Path = PRESETS) -> tuple[list[int], list[str]]:
    """Load and validate the small editable preset file."""
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("preset file must contain fps and resolutions lists")
    fps = data.get("fps")
    resolutions = data.get("resolutions")
    if not isinstance(fps, list) or not fps or any(not isinstance(value, int) or value <= 0 for value in fps):
        raise ValueError("fps must be a non-empty list of positive integers")
    if not isinstance(resolutions, list) or not resolutions or any(not isinstance(value, str) for value in resolutions):
        raise ValueError("resolutions must be a non-empty list of strings")
    for resolution in resolutions:
        if resolution != "Source" and not re.fullmatch(r"\d+x\d+", resolution):
            raise ValueError(f"invalid resolution preset: {resolution}")
    return fps, resolutions


def inspect_sequence(directory: Path) -> ImageSequence:
    """Require a contiguous, same-sized numbered JPEG sequence."""
    if not directory.is_dir():
        raise ValueError("choose an image folder")
    entries = []
    for path in directory.iterdir():
        match = IMAGE_NAME.fullmatch(path.name)
        if path.is_file() and path.suffix.lower() == ".jpg":
            if not match:
                raise ValueError(f"JPEG name is not numbered: {path.name}")
            entries.append((int(match.group(1)), match.group(1), path))
    if not entries:
        raise ValueError("folder contains no numbered .jpg images")
    entries.sort()
    numbers = [number for number, _, _ in entries]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        raise ValueError("JPEG frame numbers must be contiguous")
    digits = len(entries[0][1])
    if any(len(number_text) != digits for _, number_text, _ in entries):
        raise ValueError("JPEG frame numbers must use the same zero-padding")
    first = cv2.imread(str(entries[0][2]))
    if first is None:
        raise ValueError(f"cannot read {entries[0][2].name}")
    height, width = first.shape[:2]
    for _, _, path in entries[1:]:
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (height, width):
            raise ValueError(f"unreadable or differently sized image: {path.name}")
    return ImageSequence(directory, f"%0{digits}d.jpg", numbers[0], len(entries), width, height)


def parse_annotations(path: Path, frame_count: int) -> list[tuple[float, float, float, float]]:
    """Read UAV123 four-column text or NanoTracker five-column CSV annotations."""
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = [row for row in csv.reader(file) if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("annotation file is empty")
    header = [cell.strip().lower() for cell in rows[0]]
    boxes: list[tuple[float, float, float, float]] = []
    if {"frame", "x", "y", "width", "height"}.issubset(header):
        indexes = {name: header.index(name) for name in ("frame", "x", "y", "width", "height")}
        seen = set()
        for row in rows[1:]:
            try:
                frame = int(row[indexes["frame"]])
                box = tuple(float(row[indexes[name]]) for name in ("x", "y", "width", "height"))
            except (IndexError, ValueError) as error:
                raise ValueError("invalid NanoTracker annotation row") from error
            if frame in seen or frame < 0 or frame >= frame_count:
                raise ValueError("annotation frame numbers must be unique and zero-based")
            seen.add(frame)
            boxes.append((frame, box))
        if seen != set(range(frame_count)):
            raise ValueError("annotation frames must exactly match the image sequence")
        boxes.sort()
        values = [box for _, box in boxes]
    else:
        try:
            values = [tuple(float(cell.strip()) for cell in row) for row in rows]
        except ValueError as error:
            raise ValueError("UAV123 annotations must be x,y,width,height rows") from error
        if any(len(box) != 4 for box in values):
            raise ValueError("UAV123 annotations must have four columns")
    if len(values) != frame_count:
        raise ValueError("annotation row count must equal image-frame count")
    if any(width <= 0 or height <= 0 for _, _, width, height in values):
        raise ValueError("annotation width and height must be positive")
    return values


def output_size(sequence: ImageSequence, preset: str) -> tuple[int, int]:
    return (sequence.width, sequence.height) if preset == "Source" else tuple(map(int, preset.split("x")))


def resized_box(box: tuple[float, float, float, float], source: tuple[int, int], target: tuple[int, int]) -> tuple[float, float, float, float]:
    """Apply the same explicit scale and letterbox padding used by ffmpeg."""
    source_width, source_height = source
    target_width, target_height = target
    scale = min(target_width / source_width, target_height / source_height)
    scaled_width = max(2, round(source_width * scale / 2) * 2)
    scaled_height = max(2, round(source_height * scale / 2) * 2)
    pad_x = (target_width - scaled_width) // 2
    pad_y = (target_height - scaled_height) // 2
    x, y, width, height = box
    return x * scaled_width / source_width + pad_x, y * scaled_height / source_height + pad_y, width * scaled_width / source_width, height * scaled_height / source_height


def ffmpeg_command(sequence: ImageSequence, fps: int, target: tuple[int, int], output: Path) -> list[str]:
    command = ["ffmpeg", "-y", "-framerate", str(fps), "-start_number", str(sequence.start_number), "-i", str(sequence.directory / sequence.pattern)]
    if target != (sequence.width, sequence.height):
        scaled_width = max(2, round(sequence.width * min(target[0] / sequence.width, target[1] / sequence.height) / 2) * 2)
        scaled_height = max(2, round(sequence.height * min(target[0] / sequence.width, target[1] / sequence.height) / 2) * 2)
        pad_x = (target[0] - scaled_width) // 2
        pad_y = (target[1] - scaled_height) // 2
        command += ["-vf", f"scale={scaled_width}:{scaled_height},pad={target[0]}:{target[1]}:{pad_x}:{pad_y}:black"]
    return command + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-progress", "pipe:1", "-nostats", str(output)]


def write_annotations(path: Path, boxes: Iterable[tuple[float, float, float, float]], source: tuple[int, int], target: tuple[int, int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "x", "y", "width", "height"])
        for frame, box in enumerate(boxes):
            writer.writerow([frame, *resized_box(box, source, target)])


def write_metadata(path: Path, sequence: ImageSequence, fps: int, target: tuple[int, int], destination: Path, annotation: Path | None, sidecar: Path | None) -> None:
    """Write the conversion details beside the generated video."""
    metadata = {
        "source": {
            "image_directory": str(sequence.directory),
            "frame_count": sequence.count,
            "fps": fps,
            "resolution": {"width": sequence.width, "height": sequence.height},
        },
        "output": {
            "video": str(destination),
            "fps": fps,
            "resolution": {"width": target[0], "height": target[1]},
        },
    }
    if annotation and sidecar:
        metadata["annotation"] = {"source": str(annotation), "output": str(sidecar)}
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


class ConverterApp:
    def __init__(self, root) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.tk = tk
        self.ttk = ttk
        fps, resolutions = load_presets()
        self.source = tk.StringVar()
        self.destination = tk.StringVar()
        self.annotation = tk.StringVar()
        self.import_annotations = tk.BooleanVar()
        self.fps = tk.StringVar(value=str(fps[0]))
        self.resolution = tk.StringVar(value=resolutions[0])
        self.status = tk.StringVar(value="Choose an image folder and MP4 destination.")
        self.progress = tk.DoubleVar()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.process: subprocess.Popen[str] | None = None
        self.cancelled = False
        self.running = False

        root.title("Image sequence to MP4")
        root.columnconfigure(1, weight=1)
        self._path_row("Image folder", self.source, self.choose_source, 0, "Choose folder")
        self._path_row("MP4 destination", self.destination, self.choose_destination, 1, "Save as")
        ttk.Checkbutton(root, text="Import annotations", variable=self.import_annotations, command=self.toggle_annotations).grid(row=2, column=0, padx=8, pady=5, sticky="w")
        self.annotation_entry = ttk.Entry(root, textvariable=self.annotation, state="disabled")
        self.annotation_entry.grid(row=2, column=1, padx=8, pady=5, sticky="ew")
        self.annotation_button = ttk.Button(root, text="Choose file", command=self.choose_annotation, state="disabled")
        self.annotation_button.grid(row=2, column=2, padx=8, pady=5)
        ttk.Label(root, text="FPS").grid(row=3, column=0, padx=8, pady=5, sticky="w")
        ttk.Combobox(root, textvariable=self.fps, values=[str(value) for value in fps], state="readonly", width=12).grid(row=3, column=1, padx=8, pady=5, sticky="w")
        ttk.Label(root, text="Resolution").grid(row=4, column=0, padx=8, pady=5, sticky="w")
        ttk.Combobox(root, textvariable=self.resolution, values=resolutions, state="readonly", width=12).grid(row=4, column=1, padx=8, pady=5, sticky="w")
        ttk.Progressbar(root, variable=self.progress, maximum=100).grid(row=5, column=0, columnspan=3, padx=8, pady=(10, 4), sticky="ew")
        ttk.Label(root, textvariable=self.status).grid(row=6, column=0, columnspan=3, padx=8, pady=4, sticky="w")
        self.convert_button = ttk.Button(root, text="Convert", command=self.convert)
        self.convert_button.grid(row=7, column=1, padx=8, pady=8, sticky="e")
        self.cancel_button = ttk.Button(root, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_button.grid(row=7, column=2, padx=8, pady=8, sticky="e")

    def _path_row(self, label, variable, command, row, button_text) -> None:
        self.ttk.Label(self.root, text=label).grid(row=row, column=0, padx=8, pady=5, sticky="w")
        self.ttk.Entry(self.root, textvariable=variable).grid(row=row, column=1, padx=8, pady=5, sticky="ew")
        self.ttk.Button(self.root, text=button_text, command=command).grid(row=row, column=2, padx=8, pady=5)

    def choose_source(self) -> None:
        from tkinter import filedialog

        path = filedialog.askdirectory(title="Choose numbered JPEG folder", initialdir=DEFAULT_SOURCE_DIR)
        if path:
            self.source.set(path)
            if not self.destination.get():
                self.destination.set(str(ROOT / "data" / f"{Path(path).name}.mp4"))

    def choose_destination(self) -> None:
        from tkinter import filedialog

        path = filedialog.asksaveasfilename(title="Save MP4", initialdir=ROOT / "data", defaultextension=".mp4", filetypes=[("MP4 video", "*.mp4")])
        if path:
            self.destination.set(path)

    def toggle_annotations(self) -> None:
        state = "normal" if self.import_annotations.get() else "disabled"
        self.annotation_entry.configure(state=state)
        self.annotation_button.configure(state=state)
        if state == "disabled":
            self.annotation.set("")

    def choose_annotation(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(title="Choose annotation file", initialdir=DEFAULT_SOURCE_DIR, filetypes=[("Annotations", "*.txt *.csv"), ("All files", "*.*")])
        if path:
            self.annotation.set(path)

    def convert(self) -> None:
        from tkinter import messagebox

        if not shutil.which("ffmpeg"):
            messagebox.showerror("ffmpeg missing", "Install ffmpeg and try again.")
            return
        try:
            sequence = inspect_sequence(Path(self.source.get()))
            destination = Path(self.destination.get())
            if not destination.name:
                raise ValueError("choose an MP4 destination")
            if destination.suffix.lower() != ".mp4":
                destination = destination.with_suffix(".mp4")
                self.destination.set(str(destination))
            target = output_size(sequence, self.resolution.get())
            boxes = None
            annotation_path = None
            if self.import_annotations.get():
                if not self.annotation.get():
                    raise ValueError("choose an annotation file or clear Import annotations")
                annotation_path = Path(self.annotation.get())
                boxes = parse_annotations(annotation_path, sequence.count)
            sidecar = destination.with_suffix(".csv") if boxes is not None else None
            metadata = destination.with_suffix(".json")
            conflicts = [path for path in (destination, sidecar, metadata) if path and path.exists()]
            if conflicts and not messagebox.askyesno("Replace output?", "Replace existing output(s)?\n" + "\n".join(str(path) for path in conflicts)):
                return
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot convert", str(error))
            return
        self.start_worker(sequence, int(self.fps.get()), target, destination, boxes, annotation_path, sidecar, metadata)

    def start_worker(self, sequence, fps, target, destination, boxes, annotation, sidecar, metadata) -> None:
        self.running = True
        self.cancelled = False
        self.progress.set(0)
        self.status.set("Converting…")
        self.convert_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        thread = threading.Thread(target=self._worker, args=(sequence, fps, target, destination, boxes, annotation, sidecar, metadata), daemon=True)
        thread.start()
        self.root.after(50, self.poll)

    def _worker(self, sequence, fps, target, destination, boxes, annotation, sidecar, metadata) -> None:
        temporary_video = destination.with_name(f"{destination.stem}.part{destination.suffix}")
        temporary_csv = sidecar.with_name(f"{sidecar.stem}.part{sidecar.suffix}") if sidecar else None
        temporary_metadata = metadata.with_name(f"{metadata.stem}.part{metadata.suffix}")
        try:
            if temporary_csv and boxes is not None:
                write_annotations(temporary_csv, boxes, (sequence.width, sequence.height), target)
            write_metadata(temporary_metadata, sequence, fps, target, destination, annotation, sidecar)
            command = ffmpeg_command(sequence, fps, target, temporary_video)
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            tail = []
            assert self.process.stdout is not None
            for line in self.process.stdout:
                line = line.strip()
                if line.startswith("frame="):
                    try:
                        self.events.put(("progress", min(100, int(line.split("=", 1)[1]) * 100 / sequence.count)))
                    except ValueError:
                        pass
                elif line:
                    tail.append(line)
                    tail = tail[-5:]
            result = self.process.wait()
            if self.cancelled:
                raise RuntimeError("Conversion cancelled")
            if result:
                raise RuntimeError("ffmpeg failed:\n" + "\n".join(tail))
            os.replace(temporary_video, destination)
            if temporary_csv and sidecar:
                os.replace(temporary_csv, sidecar)
            os.replace(temporary_metadata, metadata)
            self.events.put(("done", f"Created {destination} and {metadata}"))
        except Exception as error:
            for path in (temporary_video, temporary_csv, temporary_metadata):
                if path:
                    path.unlink(missing_ok=True)
            self.events.put(("error", str(error)))
        finally:
            self.process = None

    def cancel(self) -> None:
        self.cancelled = True
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.status.set("Cancelling…")
        self.cancel_button.configure(state="disabled")

    def poll(self) -> None:
        from tkinter import messagebox

        try:
            while True:
                event, value = self.events.get_nowait()
                if event == "progress":
                    self.progress.set(value)
                else:
                    self.running = False
                    self.convert_button.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
                    self.status.set(str(value))
                    if event == "done":
                        self.progress.set(100)
                        messagebox.showinfo("Conversion complete", str(value))
                    elif value != "Conversion cancelled":
                        messagebox.showerror("Conversion failed", str(value))
                    return
        except queue.Empty:
            pass
        if self.running:
            self.root.after(50, self.poll)


def main() -> None:
    import tkinter as tk

    root = tk.Tk()
    ConverterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
