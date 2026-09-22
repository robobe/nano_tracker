# NanoTrack V3 model notice

`models/nanotrackv3/nanotrack_backbone.onnx` originates from the NanoTrack V3 model in HonglinChu/SiamTrackers NanoTrack. Its spatial input and output metadata was changed from fixed `255×255` / `16×16` to dynamic dimensions. This permits NanoTrack's required `127×127` template crop to produce an `8×8` feature map. No model weights or graph operations were changed.

`models/nanotrackv3_ncnn/` contains NCNN `.param` and `.bin` files generated from those ONNX models using PNNX. No model weights were changed.

The original project is licensed under Apache-2.0; see `NanoTrack-LICENSE`.

[← Return to README.md](../README.md)
