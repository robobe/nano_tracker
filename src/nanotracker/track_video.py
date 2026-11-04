"""Track an object in a video using the NanoTracker Python library."""

import argparse
from pathlib import Path

import cv2

import nanotracker
from nanotracker import Engine, NanoTracker, version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"NanoTracker {version()}")
    sample_video = Path(nanotracker.__file__).parent / "demo_data" / "car4.mp4"
    parser.add_argument("video", nargs="?", default=str(sample_video))
    parser.add_argument("--engine", choices=[engine.name.lower() for engine in Engine], default="onnx")
    parser.add_argument("--model-path", help="external directory containing engine model files")
    args = parser.parse_args()
    print(f"NanoTracker {version()}")
    engine = Engine[args.engine.upper()]
    capture = cv2.VideoCapture(args.video)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Cannot read video: {args.video}")
    bbox = cv2.selectROI("Select target", frame, showCrosshair=True)
    cv2.destroyWindow("Select target")
    tracker = NanoTracker(engine, args.model_path)
    tracker.init(frame, bbox)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        bbox, confidence = tracker.update(frame)
        x, y, width, height = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame, f"{confidence:.2f}", (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("NanoTracker", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
