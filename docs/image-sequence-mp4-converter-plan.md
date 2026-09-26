# Image-sequence MP4 converter with annotations

Add a Tkinter converter at `scripts/image_sequence_to_mp4.py` and YAML presets at
`scripts/video_presets.yaml`. It converts a contiguous UAV123-style JPEG sequence
to an H.264 MP4 and, when the **Import annotations** checkbox is selected, converts
UAV123 or NanoTracker annotations to a NanoTracker CSV beside that MP4.

- Presets provide FPS `10`, `20`, `30` and `Source`, `640x360`, `1280x720`
  resolutions through read-only comboboxes.
- Resized videos preserve aspect ratio with black letterboxing, and annotations are
  scaled and offset identically.
- Each completed conversion writes a JSON sidecar with source/output settings and
  optional annotation paths.
- The GUI confirms output replacement, shows progress, permits cancellation, and
  writes temporary files before replacing completed MP4/CSV outputs.
- Tests cover presets, image/annotation validation, and coordinate transformation.
