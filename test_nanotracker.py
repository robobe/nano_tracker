"""Small checks for the estimator and correction decision."""

import unittest

import numpy as np

from nanotracker import KalmanBoxEstimator, matching_annotation, should_correct


class EstimatorTests(unittest.TestCase):
    def test_prediction_starts_at_initial_box(self) -> None:
        estimator = KalmanBoxEstimator()
        estimator.initialize((10, 20, 30, 40))
        np.testing.assert_allclose(estimator.predict(), [10, 20, 30, 40])

    def test_correction_gate_respects_enablement_and_threshold(self) -> None:
        estimate = np.array([0, 0, 10, 10], dtype=np.float32)
        measurement = np.array([30, 30, 10, 10], dtype=np.float32)
        self.assertFalse(should_correct(estimate, measurement, False, 0.2))
        self.assertTrue(should_correct(estimate, measurement, True, 0.2))

    def test_matching_annotation_requires_a_sibling_csv(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as temporary:
            video = Path(temporary) / "car4.mp4"
            video.touch()
            self.assertIsNone(matching_annotation(video))
            annotation = video.with_suffix(".csv")
            annotation.touch()
            self.assertEqual(matching_annotation(video), annotation)


if __name__ == "__main__":
    unittest.main()
