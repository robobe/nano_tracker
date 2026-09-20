# C++ NanoTrack V3 example plan

Keep the Python application unchanged and add a C++20 example under `cpp/`. It shares the vendored V3 ONNX models, uses OpenCV for local-video input, ROI selection, display, and preprocessing, and uses ONNX Runtime 1.30 with CPU/CUDA selection.

The C++ app accepts `--input`, `--device auto|cpu|cuda`, and `--self-check`. The self-check runs both ONNX models against a synthetic frame. The app shows the tracked box, NanoTrack confidence, and FPS; annotation CSV, output recording, and a native file picker remain Python-only.

The build uses root CMake presets and an external `ONNXRUNTIME_ROOT` pointing to the official Linux CUDA 13 ONNX Runtime archive. OpenCV comes from the system package. CUDA 13/cuDNN 9 runtime libraries are required for GPU execution. Tests cover CMake configuration/build, self-check, CPU video tracking, and CUDA model execution.
