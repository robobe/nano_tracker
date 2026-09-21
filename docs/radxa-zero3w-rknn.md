# NanoTrack V3 on Radxa Zero 3W with RKNN

The Radxa Zero 3W uses Rockchip RK3566. This guide runs NanoTrack V3 headlessly: OpenCV reads the local video, RKNN runs the neural-network models on the NPU, and the CPU performs crop and tracking ranking.

The supplied RKNN models are static FP16 models for **RK3566**. They are not compatible with the older Radxa Zero (Amlogic) or RK3588 boards.

## 1. Install Radxa OS and enable the NPU

Install the official Radxa OS Debian 12 image for Zero 3W, boot it, then enable the RK356X NPU:

```bash
sudo rsetup
# Overlays -> Manage overlays -> Enable NPU
sudo reboot
sudo apt update
sudo apt install rknpu2-rk356x python3-opencv python3-numpy \
  build-essential cmake ninja-build libopencv-dev git
sudo dmesg | grep 'Initialized rknpu'
```

The final command must print an `rknpu` line. If it does not, run **System -> System Update** in `rsetup`, reboot, and enable the NPU again.

## 2. Convert the models on an x86 Ubuntu host

Do this once on an x86_64 Ubuntu 24.04 machine with Python 3.12; do **not** convert on the Zero 3W. The RKNN conversion environment is separate from this project's normal `uv` environment.

```bash
git clone --branch v2.3.2 https://github.com/airockchip/rknn-toolkit2.git ~/rknn-toolkit2
cd ~/rknn-toolkit2/rknn-toolkit2/packages/x86_64
uv venv --python 3.12 ~/rknn-venv
uv pip install --python ~/rknn-venv/bin/python -r requirements_cp312-2.3.2.txt
uv pip install --python ~/rknn-venv/bin/python \
  'setuptools<81' 'onnx==1.16.1' onnxruntime \
  rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
cd /path/to/nano_tracker
~/rknn-venv/bin/python scripts/convert_rknn_models.py
```

The repository includes these converted files. Re-run conversion only after changing the source ONNX models, then commit the replacements for offline deployment:

```text
models/nanotrackv3_rknn_rk3566/backbone_template.rknn
models/nanotrackv3_rknn_rk3566/backbone_search.rknn
models/nanotrackv3_rknn_rk3566/head.rknn
```

The converter compares RKNN simulator outputs with ONNX Runtime and prints the maximum difference. It fails on a wrong tensor shape or non-finite output. Use `--no-validate` only to diagnose a simulator problem.

## 3. Install the Python runner on Zero 3W

Copy this repository and the `rknn-toolkit2` checkout to the board. The Lite2 wheel must match the board's Python version. Radxa OS Debian 12 uses Python 3.11:

```bash
cd /path/to/nano_tracker
uv venv --system-site-packages --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python \
  /path/to/rknn-toolkit2/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
.venv/bin/python nanotracker.py --engine rknn --self-check
.venv/bin/python nanotracker.py --engine rknn --input data/video.mp4 \
  --roi 100,100,80,60 --no-display
```

`--no-display` intentionally requires a local video and an initial `x,y,width,height` ROI. It does not use Tk or OpenCV windows, so it works through SSH.

## 4. Build and run the C++ runner on Zero 3W

The C++ API header and `librknnrt.so` come from the same Toolkit2 checkout. Build natively on the board:

```bash
cd /path/to/nano_tracker
export RKNN_ROOT=/path/to/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api
cmake --preset debug-rknn
cmake --build --preset debug-rknn
./build/debug-rknn/cpp/nanotracker_rknn --self-check
./build/debug-rknn/cpp/nanotracker_rknn --input data/video.mp4 \
  --roi 100,100,80,60 --no-display
```

Both runners print the average tracking FPS. Neither saves an output video.

## Troubleshooting

- **`rknpu` is absent:** use an official Zero 3W image, enable the NPU in `rsetup`, then reboot. `rknpu2-rk356x` is the correct package for RK3566.
- **`No module named rknnlite`:** install the ARM64 Lite2 wheel matching `python3 --version`; do not install the x86 Toolkit2 wheel on the board.
- **`librknnrt.so` error:** keep `RKNN_ROOT` pointed at the Toolkit2 `librknn_api` directory while configuring/building C++, and use a Runtime/Lite2 version compatible with the model's Toolkit2 version.
- **Slow or inaccurate tracker:** first confirm FP16 works. INT8 needs representative template and search crops for calibration; do not quantize the matching head blindly.

Sources: [Radxa Zero 3 documentation](https://docs.radxa.com/en/zero/zero3), [Radxa RKNN installation](https://docs.radxa.com/en/rock5/rock5b/app-development/ai/rknn-install), and [Rockchip RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2).
