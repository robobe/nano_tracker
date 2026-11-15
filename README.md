# NanoTracker

NanoTrack V3 single-object tracking library and local-video demos for Python. Supports RK3566 NPU inference and ONNX Runtime CPU/CUDA inference.

## Overview

Initialize a target from a box in the first video frame, then track its position and confidence in subsequent frames. The library is intended for Python applications processing OpenCV images; the demos provide interactive video playback and ROI selection. Tracking runs locally once dependencies and models are installed.

## Features

- A small API: `NanoTracker.init()` and `update()`, with `BoundingBox` results.
- RK3566 inference using bundled static FP16 RKNN models; ONNX CPU and CUDA engines using the included ONNX models.
- Interactive video demos and an optional annotation/IoU workflow.
- Numbered JPEG sequence conversion to H.264 MP4, with annotation scaling and metadata sidecars.

## Architecture

`NanoTracker` validates inputs and selects an inference engine. `TrackerCore` handles cropping, model matching, and box updates; engine classes handle model loading and inference. Demos and conversion tools sit outside the tracking core.

## Requirements

- Python **3.10+**; `uv` for the commands below.
- Base dependencies: NumPy, OpenCV, and PyYAML (installed by `uv sync`).
- RK3566: a compatible board, enabled NPU/driver, and a matching ARM64 RKNN Toolkit Lite2 wheel.
- ONNX CPU: separately installed `onnxruntime`. CUDA requires a compatible ONNX Runtime GPU installation and NVIDIA driver.
- Interactive demos require a graphical display and OpenCV GUI support. The annotation GUI and sequence converter also need Tkinter; the converter needs FFmpeg with H.264 encoding support.

## Installation

Run from the repository root:

```bash
uv sync
```

This installs the library and base dependencies. The standard wheel bundles RK3566 models; inference runtimes are installed separately.

For CPU inference from this checkout:

```bash
uv pip install onnxruntime
```

For RK3566, enable the NPU and install the board driver (`rknpu2-rk356x` on Radxa Zero 3W), then install the Lite2 wheel matching your board's Python version:

```bash
uv pip install /path/to/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

This wheel name is an example for ARM64 Python 3.11. Obtain the matching wheel from [Rockchip Toolkit2](https://github.com/airockchip/rknn-toolkit2). Use `uv run --no-sync` after separately installing a runtime to preserve it.

## Quick Start

With the CPU runtime installed, pass an existing local video explicitly:

```bash
uv run --no-sync nanotracker-demo /path/to/video.mp4 --engine onnx
```

Select the target ROI and confirm it in the OpenCV window. Playback displays a green tracking box and confidence. Press `q` or Escape to exit. The preview is not recorded.

On a configured RK3566 board, use `--engine rk3566` instead.

## Usage

### Library API

Images are non-empty `HxWx3` NumPy arrays in OpenCV BGR order. Boxes use `(x, y, width, height)` pixel coordinates with positive width and height. `update()` returns a `BoundingBox` named tuple and confidence; use `np.asarray(bbox)` if an array is needed.

```python
import cv2
from nanotracker import BoundingBox, Engine, NanoTracker

video = cv2.VideoCapture("/path/to/video.mp4")
try:
    ok, first_frame = video.read()
    if not ok:
        raise RuntimeError("Could not read the video")

    tracker = NanoTracker(engine=Engine.ONNX)
    # Replace these coordinates with the target box in the first frame.
    tracker.init(first_frame, BoundingBox(x=100, y=80, width=60, height=45))
    while True:
        ok, frame = video.read()
        if not ok:
            break
        bbox, confidence = tracker.update(frame)
        print(f"Box: {bbox}, confidence: {confidence:.2f}")
finally:
    video.release()
```

`NanoTracker()` defaults to `Engine.RK3566` in this checkout. Other implemented choices are `Engine.ONNX` and `Engine.ONNX_CUDA`. `Engine.NCNN` is reserved and raises `EngineUnavailableError`; the separate NCNN classification tutorial does not implement a tracker engine.

Use `nanotracker.version()` or `NanoTracker.version()` to read the installed version. Catch `NanoTrackerError` for library errors, including missing models, unavailable engines, invalid images, tracking before initialization, and inference failures. Invalid boxes raise `ValueError`; a constructor engine that is not an `Engine` value raises `TypeError`.

### Annotation GUI

The separate ONNX GUI supports selecting a video, drawing an ROI, optional annotation CSV, IoU, and estimator controls:

```bash
uv run --no-sync python examples/nanotracker_gui.py --device cpu
```

Annotation CSV uses the header `frame,x,y,width,height` and zero-based frame numbers. Missing frame rows are allowed. See [Python usage](docs/usage.md) for workflow details; its older root-level commands and automatic GPU dependency claim are outdated. Use the paths and runtime installation instructions in this README.

### Convert an image sequence to MP4

```bash
uv run python scripts/image_sequence_to_mp4.py
```

Choose a contiguous numbered JPEG folder, destination, FPS, and resolution. `Source` keeps the source dimensions; other resolutions preserve the aspect ratio with black letterboxing. Output is H.264 MP4.

Check **Import annotations** to load UAV123 `x,y,width,height` text rows or NanoTracker `frame,x,y,width,height` CSV. The converter writes a matching `<video-name>.csv`, with boxes scaled and letterboxed to match the video. Each input image must have exactly one annotation row when annotations are imported.

Every conversion writes `<video-name>.json` with the source folder, frame count, FPS, source/output resolutions, and annotation paths when selected.

## Configuration

| Setting | Default | Purpose |
|---|---|---|
| Demo `--engine` | `rk3566` | Select `rk3566`, `onnx`, or `onnx_cuda`; `ncnn` is not implemented. |
| Demo `--model-path` / API `model_path` | Bundled or checkout models | Override the engine model directory. |
| Demo positional `video` | Package-relative `demo_data/car4.mp4` | Pass an existing video explicitly; the default sample is not bundled by the current build configuration. |
| GUI `--device` | `auto` | Select ONNX CPU/CUDA, with CPU fallback in auto mode. |
| Converter presets | `scripts/video_presets.yaml` | FPS: 10, 20, 30; resolutions: `Source`, `640x360`, `1280x720`. |

ONNX model directories must contain `nanotrack_backbone.onnx` and `nanotrack_head.onnx`. RK3566 directories must contain `backbone_template.rknn`, `backbone_search.rknn`, and `head.rknn`. ONNX models are available under `models/nanotrackv3` in the checkout but are not included in the standard RK3566 wheel.

## Project Structure

```text
src/nanotracker/  Library, tracking core, engines, and packaged demo
examples/        Video entry point and annotation GUI
scripts/         Conversion, CUDA checks, and CPU wheel build tools
models/          ONNX and RK3566 model files
ncnn_demo/       Standalone NCNN classification tutorial
tests/          Library API tests
docs/           Guides and saved implementation plans
```

## Development

Use `uv sync` for the development environment. Package version **0.1.0** and dependencies are declared in `pyproject.toml`; Hatchling builds the package.

```bash
uv build
```

To build a separate CPU wheel with ONNX models, the CPU runtime dependency, and ONNX defaults:

```bash
uv run python scripts/build_cpu_wheel.py
```

The CPU wheel is written to `dist/cpu/`. Formatting, linting, and a dedicated debugging workflow are unspecified in the project configuration.

## Testing

Run the API tests, converter tests, and GUI calculation self-check:

```bash
uv run python -m unittest discover -s tests
uv run python scripts/test_image_sequence_to_mp4.py
uv run python examples/nanotracker_gui.py --self-check
```

These checks do not validate real model inference or RK3566 hardware. With a separately configured CUDA runtime, use `uv run --no-sync python scripts/check_cuda.py` to check actual CUDA model execution.

## Documentation

- [Python usage](docs/usage.md) — annotation GUI workflow; see the command/dependency corrections above.
- [How NanoTrack works](docs/how-nanotrack-works.md) — models and matching explained.
- [NCNN Python tutorial](docs/ncnn-python-tutorial.md) — standalone image classification example.
- [Radxa Zero 3W RKNN setup](docs/radxa-zero3w-rknn.md) — NPU setup and model conversion. Its legacy `nanotracker_gui.py --engine rknn` runner commands do not match the current entry point; use `nanotracker-demo /path/to/video.mp4 --engine rk3566` with a display. The packaged demo has no headless option.

### Saved plans

Plans record design work and do not guarantee that every item is implemented.

- [Library plan](docs/library-plan.md)
- [RK3566 library plan](docs/rk3566-library-plan.md)
- [Version API plan](docs/version-api-plan.md)
- [Library exceptions plan](docs/library-exceptions-plan.md)
- [NCNN Python tutorial plan](docs/ncnn-python-tutorial-plan.md)
- [Radxa Zero 3W RKNN plan](docs/radxa-zero3w-rknn-plan.md)
- [Image-sequence converter plan](docs/image-sequence-mp4-converter-plan.md)
- [Kalman-estimator GUI plan](docs/kalman-estimator-gui-plan.md)

## Troubleshooting

- **Video cannot be read:** pass an existing video path explicitly and check that OpenCV can decode it.
- **Missing model files:** check the engine-specific filenames above and pass `--model-path` or `model_path` for an external directory.
- **Runtime unavailable:** install the runtime for the selected engine. On RK3566, check the NPU driver and matching Lite2 wheel. Use `--no-sync` when running with separately installed runtime packages.
- **CUDA unavailable:** check the NVIDIA driver and runtime compatibility. Do not install CPU and GPU ONNX Runtime wheels together. Run `scripts/check_cuda.py` to verify execution.
- **GUI cannot open:** run in a graphical session with OpenCV GUI support; the annotation GUI and converter additionally need Tkinter.

## Roadmap

No committed release roadmap is specified. Saved plans are linked above for design context.

## Contributing

A contribution policy is unspecified. Include reproduction steps for bugs and run the relevant checks before submitting changes.

## License

A project-wide license is unspecified. The vendored NanoTrack V3 models come from [HonglinChu/SiamTrackers NanoTrack](https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack), licensed under Apache-2.0. Its license and model-change notice are in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES/).
