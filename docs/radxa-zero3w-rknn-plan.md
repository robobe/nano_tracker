# RKNN deployment plan: Radxa Zero 3W

Target the Radxa Zero 3W's RK3566 NPU with three static FP16 NanoTrack V3 models: template backbone (`1x3x127x127`), search backbone (`1x3x255x255`), and matching head (`1x96x8x8` plus `1x96x16x16`). OpenCV keeps video decoding and image crops on CPU; RKNN only runs neural-network inference.

Use RKNN Toolkit2 2.3.2 on an x86 Ubuntu 24.04/Python 3.12 host to convert and validate the ONNX models. The board uses RKNN Lite2 for Python and RKNN Runtime for C++. Build and test FP16 before considering INT8 calibration.

Add a headless Python `--engine rknn` mode and a separate native C++ `nanotracker_rknn` executable. Both accept `--input`, `--roi x,y,width,height`, and `--no-display`, perform a real model self-check, and print average FPS. Preserve the existing ONNX and NCNN runners as fallback paths.

Keep generated RK3566 artifacts under `models/nanotrackv3_rknn_rk3566/` and commit them after successful conversion so the deployed project remains offline. Document Radxa OS Debian 12 installation, NPU enablement, Python setup, C++ build, and driver/runtime troubleshooting in `docs/radxa-zero3w-rknn.md`.
