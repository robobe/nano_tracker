"""Build an ONNX CPU wheel without changing the RK3566 checkout."""

import shutil
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="nanotracker-cpu-") as directory:
    stage = Path(directory)
    shutil.copytree(root / "src", stage / "src")
    shutil.copytree(root / "models" / "nanotrackv3", stage / "models" / "nanotrackv3")
    config = (root / "pyproject.toml").read_text()
    config = config.replace('    "numpy>=1.26",', '    "numpy>=1.26",\n    "onnxruntime>=1.19",')
    config = config.replace("nanotrackv3_rknn_rk3566", "nanotrackv3")
    (stage / "pyproject.toml").write_text(config)
    tracker = stage / "src/nanotracker/tracker.py"
    tracker.write_text(tracker.read_text().replace("engine: Engine = Engine.RK3566", "engine: Engine = Engine.ONNX"))
    demo = stage / "src/nanotracker/track_video.py"
    demo.write_text(demo.read_text().replace('default="rk3566"', 'default="onnx"'))
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(root / "dist/cpu"), str(stage)], check=True)
