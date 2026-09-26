# Kalman-estimator tracker UI plan

Replace the default multi-window flow with one Tkinter screen for video selection,
annotation selection, ROI drawing, preview, and tracker controls. Add a
constant-velocity Kalman box estimator whose predicted box can be displayed and,
when enabled, used to reset NanoTracker when prediction-versus-measurement IoU is
below 0.20. Show tracker, annotation, and estimate boxes through three default-on
checkboxes; auto-select a sibling annotation CSV when present. The white ROI is
removed when tracking begins. Warm the requested ONNX Runtime device in the
background at GUI startup and retain the existing headless command-line workflow.
