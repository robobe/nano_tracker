# NanoTracker

An offline, live-preview Python application for tracking one object in a local video with NanoTrack V3. It uses OpenCV for the dialogs, ROI selection, display, and preprocessing; ONNX Runtime runs the two ONNX models on CUDA when available or CPU otherwise.

## Install (Ubuntu)

Install the system tools used by the video converter and its GUI, then install the Python environment with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
sudo apt update
sudo apt install ffmpeg python3-tk
uv sync
```

> [!TIP]
> You do not need to create a virtual environment yourself. `uv sync` creates `.venv` when needed and installs the locked Python dependencies into it.

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

## Convert an image sequence to MP4

Convert a UAV123-style folder of contiguous numbered JPEG images into an MP4 for NanoTracker:

```bash
uv run python scripts/image_sequence_to_mp4.py
```

Choose the image folder and MP4 destination, then select an FPS and resolution from the editable presets in `scripts/video_presets.yaml`. `Source` keeps the original image dimensions; other resolutions preserve the aspect ratio with black letterboxing. The converter creates H.264 MP4 files.

Check **Import annotations** to select an optional annotation file. It accepts either UAV123 `x,y,width,height` text rows or NanoTracker `frame,x,y,width,height` CSV, and writes a matching `<video-name>.csv` beside the MP4. Its boxes are automatically scaled and letterboxed to match the converted video. Every conversion also writes `<video-name>.json`, recording the source folder, frame count, selected FPS, source/output resolutions, and annotation paths when selected. The sequence and annotation must each have exactly one frame/row per image.

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

For a minimal image-classification example using NCNN's Python API, see [the Python NCNN tutorial](docs/ncnn-python-tutorial.md).

Want to understand the tracker itself? Read [How NanoTrack works](docs/how-nanotrack-works.md).

## Radxa Zero 3W RKNN example

The headless RK3566 NPU example uses RKNN with static FP16 NanoTrack V3 models. See [Radxa Zero 3W RKNN usage](docs/radxa-zero3w-rknn.md) for host conversion, device setup, and Python/C++ commands.

## Documentation

- [Python usage](docs/usage.md) — GUI workflow, annotations, and ONNX Runtime CPU/CUDA setup.
- [C++ usage](docs/cpp-usage.md) — build and run the ONNX Runtime and NCNN C++ examples.
- [How NanoTrack works](docs/how-nanotrack-works.md) — a step-by-step explanation of the model and matching process.
- [NCNN Python tutorial](docs/ncnn-python-tutorial.md) — minimal NCNN inference with a pretrained image model.
- [Radxa Zero 3W RKNN usage](docs/radxa-zero3w-rknn.md) — convert models and run headless Python/C++ NPU tracking.

### Saved plans

- [C++ example plan](docs/cpp-example-plan.md) — add the original ONNX Runtime C++ tracker.
- [C++ CUDA runtime plan](docs/cpp-cuda-runtime-plan.md) — make C++ CUDA provider libraries discoverable.
- [NCNN CPU plan](docs/ncnn-cpu-plan.md) — convert NanoTrack V3 and run it through NCNN CPU.
- [NCNN Python tutorial plan](docs/ncnn-python-tutorial-plan.md) — add the standalone NCNN Python learning example.
- [Radxa Zero 3W RKNN plan](docs/radxa-zero3w-rknn-plan.md) — deploy static RK3566 models with headless Python and C++ runners.
- [Image-sequence converter plan](docs/image-sequence-mp4-converter-plan.md) — convert UAV123 JPEG sequences and annotations to MP4/CSV.

## Attribution

The vendored NanoTrack V3 models come from [HonglinChu/SiamTrackers NanoTrack](https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack), licensed under Apache-2.0. Its license and model-change notice are in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES/).
