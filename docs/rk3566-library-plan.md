# RK3566 library engine plan

Add an RK3566 engine using RKNN Lite2 and the existing fixed-shape template, search, and head models. Build the wheel with those RK3566 models instead of ONNX models, and make the RK3566 engine the default. Keep the existing external model directory option.

Install NumPy and OpenCV through project dependencies; document installation of the board-specific Lite2 wheel and NPU driver separately. Check that the wheel contains all three RKNN files and test initialization and inference on RK3566 hardware.
