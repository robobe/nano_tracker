# C++ CUDA runtime loading plan

The C++ executable finds ONNX Runtime through CMake RPATH, but its CUDA provider needs CUDA 13 and cuDNN 9 runtime libraries. The default solution is `scripts/run_cpp.sh`, which discovers the NVIDIA libraries installed by `uv sync` and supplies them only to the launched process.

The launcher forwards all executable arguments, validates that the debug binary and venv libraries exist, and reports clear setup errors. README and usage documentation use it for CPU and CUDA checks as well as interactive tracking.

An optional Ubuntu 24.04 system-wide setup enables NVIDIA's CUDA package repository, installs CUDA Toolkit 13.4 and cuDNN 9 for CUDA 13, and refreshes the loader cache. That route allows direct execution without the launcher. Validation covers the direct failure before the loader path, launcher CPU success, launcher CUDA success, and direct CUDA success after a system-wide setup.
