# NanoTracker

An offline, live-preview Python application for tracking one object in a local video with NanoTrack V3. It uses OpenCV for the dialogs, ROI selection, display, and preprocessing; ONNX Runtime runs the two ONNX models on CUDA when available or CPU otherwise.

## Run

```bash
uv sync
uv run python nanotracker.py
```

Choose a video, optionally choose an annotation CSV, then drag the initial object box. Green is the tracked box and its NanoTrack confidence. When annotations exist, blue is the ground-truth box and its IoU with the tracked box. Press `q` or `Esc` to quit. Nothing is saved to disk.

Use `--device cpu`, `--device cuda`, or the default `--device auto`. `auto` selects CUDA when it is available and otherwise uses CPU.

Check that both models really execute on CUDA:

```bash
uv run python check_cuda.py
```

The model files are included, so running needs no network after `uv sync` has installed dependencies. See [docs/usage.md](docs/usage.md) for annotation format and GPU requirements.

## C++ example

The C++20 example has the same tracking model and OpenCV ROI/display loop, but uses a command-line video path:

```bash
curl -LO https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-x64-gpu_cuda13-1.30.0.tgz
tar -xzf onnxruntime-linux-x64-gpu_cuda13-1.30.0.tgz
export ONNXRUNTIME_ROOT="$PWD/onnxruntime-linux-x64-gpu_cuda13-1.30.0"
cmake --preset debug
cmake --build --preset debug
./scripts/run_cpp.sh --self-check --device cuda
./scripts/run_cpp.sh --input data/your-video.mp4 --device auto
```

The launcher supplies the CUDA 13/cuDNN 9 libraries installed by `uv sync` to the C++ process. See [C++ usage](docs/cpp-usage.md) for the complete build, system-wide installation, test, and run commands.

## C++ NCNN CPU example

The NCNN example runs the same V3 tracker on CPU and requires neither CUDA nor ONNX Runtime:

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build libopencv-dev curl unzip
curl -LO https://github.com/Tencent/ncnn/releases/download/20260526/ncnn-20260526-ubuntu-2404-shared.zip
unzip ncnn-20260526-ubuntu-2404-shared.zip
export NCNN_ROOT="$PWD/ncnn-20260526-ubuntu-2404-shared"
cmake --preset debug-ncnn
cmake --build --preset debug-ncnn
./build/debug-ncnn/cpp/nanotracker_ncnn --self-check
./build/debug-ncnn/cpp/nanotracker_ncnn --input data/your-video.mp4
```

Select the object in the ROI window, then press Enter or Space. Press `q` or Esc to quit. The NCNN models are included; see [C++ usage](docs/cpp-usage.md#ncnn-cpu-example) for conversion and troubleshooting.

## Attribution

The vendored NanoTrack V3 models come from [HonglinChu/SiamTrackers NanoTrack](https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack), licensed under Apache-2.0. Its license and model-change notice are in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES/).
