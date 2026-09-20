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

## Attribution

The vendored NanoTrack V3 models come from [HonglinChu/SiamTrackers NanoTrack](https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack), licensed under Apache-2.0. Its license and model-change notice are in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES/).
