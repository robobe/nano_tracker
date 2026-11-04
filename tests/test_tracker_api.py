import unittest
from unittest.mock import Mock

import numpy as np

from nanotracker import BoundingBox, NanoTracker


class TrackerApiTests(unittest.TestCase):
    def setUp(self):
        self.tracker = NanoTracker.__new__(NanoTracker)
        self.tracker._core = Mock()
        self.image = np.zeros((20, 20, 3), dtype=np.uint8)

    def test_invalid_boxes_do_not_initialize(self):
        for box in [(0, 0, 0, 1), (0, 0, 1, -1), (float("nan"), 0, 1, 1), (0, 0, 1)]:
            with self.subTest(box=box), self.assertRaises(ValueError):
                self.tracker.init(self.image, box)
        self.tracker._core.initialize.assert_not_called()

    def test_valid_box_and_returned_box(self):
        self.tracker.init(self.image, (1, 2, 3, 4))
        self.tracker._core.initialize.assert_called_once_with(self.image, BoundingBox(1, 2, 3, 4))
        self.tracker._core.template = object()
        self.tracker._core.track.return_value = (np.array([1, 2, 3, 4]), 0.75)
        box, confidence = self.tracker.update(self.image)
        self.assertIsInstance(box, BoundingBox)
        self.assertEqual((box.x, box.width, confidence), (1, 3, 0.75))


if __name__ == "__main__":
    unittest.main()
