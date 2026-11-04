# NanoTracker Python library plan

Provide a small `uv`-installable Python library with a stable tracker API, selectable inference engines, bundled and external model paths, and a runnable PC video example. Keep the existing GUI script available during the transition.

The public API exports `NanoTracker` and the `Engine` enum (`ONNX`, `ONNX_CUDA`, `NCNN`). `init(image, bbox)` initializes from a BGR image and `(x, y, width, height)` box. `update(image)` returns `(bbox, confidence)`. Engine implementations own model loading and inference; shared tracking math remains in the core.

Wheel packaging includes the ONNX model directory, the `examples/track_video.py` demo, and `car4.mp4` with its matching annotation CSV. The `nanotracker-demo` command runs that packaged example and sample by default. Project guidance requires SOLID principles at module and engine boundaries.

Acceptance: the library imports after `uv sync`; ONNX CPU initializes bundled or external ONNX files; CUDA selects and requires the ONNX Runtime CUDA provider; the demo accepts video, engine, and model path and defaults to the packaged sample. NCNN Python inference is not part of this first implementation and must report that limitation clearly.
