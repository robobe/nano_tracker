# NanoTrack V3 RKNN models for RK3566

The committed `backbone_template.rknn`, `backbone_search.rknn`, and `head.rknn` files were generated on an x86 host with RKNN Toolkit2 2.3.2:

```bash
python scripts/convert_rknn_models.py
```

They are static FP16 models for Radxa Zero 3W (RK3566). Re-run conversion only after replacing the source ONNX models, then commit the regenerated files so a deployed checkout works offline.

[← Return to README.md](../../README.md)
