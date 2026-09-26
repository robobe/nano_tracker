"""Small checks for image-sequence conversion helpers."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from image_sequence_to_mp4 import ImageSequence, inspect_sequence, parse_annotations, resized_box, write_annotations, write_metadata


class ConverterTests(unittest.TestCase):
    def test_sequence_and_uav_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for frame in range(1, 3):
                self.assertTrue(cv2.imwrite(str(directory / f"{frame:06d}.jpg"), np.zeros((10, 20, 3), np.uint8)))
            (directory / "annotations.txt").write_text("1,2,3,4\n5,6,7,8\n", encoding="utf-8")
            sequence = inspect_sequence(directory)
            self.assertEqual((sequence.start_number, sequence.count, sequence.width, sequence.height), (1, 2, 20, 10))
            self.assertEqual(parse_annotations(directory / "annotations.txt", 2)[1], (5.0, 6.0, 7.0, 8.0))

    def test_letterbox_box_mapping(self) -> None:
        self.assertEqual(resized_box((10, 20, 30, 40), (100, 100), (200, 100)), (60.0, 20.0, 30.0, 40.0))

    def test_nanotracker_csv_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            annotations = directory / "annotations.csv"
            annotations.write_text("frame,x,y,width,height\n0,1,2,3,4\n1,5,6,7,8\n", encoding="utf-8")
            boxes = parse_annotations(annotations, 2)
            output = directory / "output.csv"
            write_annotations(output, boxes, (10, 10), (20, 10))
            self.assertEqual(output.read_text(encoding="utf-8").splitlines()[1], "0,6.0,2.0,3.0,4.0")

    def test_metadata_records_annotation_when_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            sequence = ImageSequence(directory, "%06d.jpg", 1, 2, 20, 10)
            metadata_path = directory / "output.json"
            write_metadata(metadata_path, sequence, 10, (640, 360), directory / "output.mp4", directory / "annotations.txt", directory / "output.csv")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["source"]["resolution"], {"width": 20, "height": 10})
            self.assertEqual(metadata["output"]["fps"], 10)
            self.assertEqual(metadata["annotation"]["output"], str(directory / "output.csv"))


if __name__ == "__main__":
    unittest.main()
