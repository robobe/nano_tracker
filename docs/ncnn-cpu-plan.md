# NCNN CPU NanoTrack V3 plan

Use PNNX to convert the vendored V3 ONNX backbone and head to four committed NCNN model files. Keep PNNX out of runtime dependencies and ignore its intermediate outputs.

Add a separate `nanotracker_ncnn` C++20 executable with the existing local-video command line, OpenCV ROI/display loop, NanoTrack preprocessing and postprocessing, and CPU NCNN inference. Keep ONNX Runtime/CUDA independent through a dedicated `debug-ncnn` preset and `NCNN_ROOT` external dependency.

Verify the build and CPU self-check, then track a local video. Document the tested Ubuntu x64 NCNN archive install and explicitly defer Raspberry Pi and Vulkan support.
