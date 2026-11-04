# NanoTracker library exceptions plan

Expose a stable `NanoTrackerError` base and focused exceptions for missing models, unavailable engines, invalid images, use before initialization, and inference failures. Preserve the underlying runtime cause with exception chaining.

Model lookup raises `ModelNotFoundError`; unavailable CUDA or unimplemented NCNN raises `EngineUnavailableError`; malformed images raise `InvalidImageError`; premature updates raise `TrackerNotInitializedError`; and ONNX execution failures raise `InferenceError`. Specific exceptions also inherit an appropriate built-in category where useful.

Acceptance: all public exception classes import from `nanotracker`, errors identify the failed operation, and wrapped runtime failures retain their original cause. Tracking results and the `(bbox, confidence)` return shape stay unchanged.
