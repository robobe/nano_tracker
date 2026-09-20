# C++ example

The C++ example shares `models/nanotrackv3/` with the Python app. It accepts a local video path, asks OpenCV for the initial ROI, then shows the tracked box, confidence, and FPS.

## Dependencies

- Ubuntu OpenCV development package (`libopencv-dev`)
- CMake, Ninja, and a C++20 compiler
- Official ONNX Runtime 1.30 Linux x64 CUDA 13 archive
- NVIDIA driver; CUDA 13 and cuDNN 9 runtime libraries for `--device cuda`

Download ONNX Runtime outside this repository:

```bash
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-x64-gpu_cuda13-1.30.0.tgz
tar -xzf onnxruntime-linux-x64-gpu_cuda13-1.30.0.tgz
export ONNXRUNTIME_ROOT="$PWD/onnxruntime-linux-x64-gpu_cuda13-1.30.0"
```

`uv sync` installs the CUDA/cuDNN runtime libraries used by the Python app. The C++ launcher discovers them and sets the dynamic-library path for that process:

```bash
./scripts/run_cpp.sh --self-check --device cuda
```

## System-wide CUDA runtime (Ubuntu 24.04)

This optional path lets the C++ executable run directly, without the launcher. Your working NVIDIA driver should be kept; do not install a driver package as part of this setup.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-13-4 cudnn9-cuda-13
sudo ldconfig
```

Then verify direct CUDA execution:

```bash
./build/debug/cpp/nanotracker_cpp --self-check --device cuda
```

NVIDIA documents the [Ubuntu CUDA repository setup](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) and the [cuDNN CUDA 13 package](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html). CUDA 13 and cuDNN 9 match the ONNX Runtime 1.30 GPU archive used by this project.

## Build and test

```bash
cmake --preset debug
cmake --build --preset debug
./scripts/run_cpp.sh --self-check --device cpu
./scripts/run_cpp.sh --self-check --device cuda
```

The self-check validates IoU and runs the backbone and head on a synthetic frame. It reports the selected provider.

## Run

```bash
./scripts/run_cpp.sh --input data/your-video.mp4 --device auto
```

Choose the target with the OpenCV ROI window, then press Enter or Space. Press `q` or `Esc` to quit.
