# Usage

Run the application with `uv run python nanotracker.py`. It first opens a native file chooser for a local video. The second dialog lets you skip or select an annotation CSV. OpenCV then displays the first frame; draw the target ROI and press Enter or Space to begin.

The preview is not recorded. Press `q` or `Esc` to close it.

## Devices

`--device auto` is the default: CUDA is chosen when ONNX Runtime can initialize `CUDAExecutionProvider`; otherwise CPU is used. Use `--device cpu` to force CPU or `--device cuda` to require CUDA.

`uv sync` installs `onnxruntime-gpu[cuda,cudnn]`: ONNX Runtime plus the matching NVIDIA CUDA and cuDNN runtime libraries. It does not install an NVIDIA driver. A supported NVIDIA driver and GPU are the only system-level GPU prerequisites; no CUDA compiler/toolkit is needed for this application.

The current `onnxruntime-gpu` releases (1.27+) use CUDA 13 and cuDNN 9. If your computer deliberately uses CUDA 12, pin `onnxruntime-gpu` to a 1.21–1.26 release instead; those releases use CUDA 12.8 and cuDNN 9. Do not install CPU and GPU ONNX Runtime wheels together.

Verify actual model execution after `uv sync`:

```bash
uv run python check_cuda.py
```

Success reports CUDA nodes for both `nanotrack_backbone.onnx` and `nanotrack_head.onnx`. A provider merely appearing in `ort.get_available_providers()` is not proof that its libraries loaded.

## Optional annotation CSV

IoU needs ground truth. Supply a CSV with this exact header and use zero-based video frame numbers:

```csv
frame,x,y,width,height
0,244,161,74,70
1,246,162,74,70
```

Rows may omit frames. The application only draws a blue ground-truth box and IoU when a row exists for the displayed frame. Without annotations it still shows the green tracker box and NanoTrack confidence.

## Check

```bash
uv run python nanotracker.py --self-check
```

This validates the IoU calculation without opening a UI or loading the models.

## C++ application

After following the [C++ dependency setup](cpp-usage.md), build and run the C++ tracker with a local video path:

```bash
cmake --preset debug
cmake --build --preset debug
./scripts/run_cpp.sh --input data/your-video.mp4 --device auto
```

It opens an OpenCV ROI window, then displays the tracked box, confidence, and FPS. Use `--device cpu` to force CPU or `--device cuda` to require CUDA. Run `./scripts/run_cpp.sh --self-check --device cuda` to verify C++ CUDA inference. See [C++ usage](cpp-usage.md) for system-wide CUDA installation.

## C++ NCNN CPU application

The NCNN C++ app is a CPU-only alternative that needs neither ONNX Runtime nor CUDA. After installing NCNN and setting `NCNN_ROOT` as described in [C++ usage](cpp-usage.md#ncnn-cpu-example), run:

```bash
cmake --preset debug-ncnn
cmake --build --preset debug-ncnn
./build/debug-ncnn/cpp/nanotracker_ncnn --self-check
./build/debug-ncnn/cpp/nanotracker_ncnn --input data/your-video.mp4
```

It uses the committed models in `models/nanotrackv3_ncnn/`, asks OpenCV to select an ROI, and displays the tracked box, confidence, and FPS. Press `q` or Esc to quit.
