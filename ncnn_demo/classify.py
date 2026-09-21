#!/usr/bin/env python3
"""Classify one image with the NCNN Python API."""

# Flow:
# image -> resize to 227x227 -> BGR ncnn.Mat -> subtract mean -> inference
# -> ImageNet scores -> print the top class.

import argparse
import sys
from pathlib import Path

import cv2
import ncnn
import numpy as np


ASSETS = Path(__file__).resolve().parent / "assets"
# This binary NCNN parameter file identifies input and output blobs by index.
INPUT_BLOB = 0
OUTPUT_BLOB = 82
IMAGE_SIZE = 227
MEAN = [104.0, 117.0, 123.0]


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run: uv run --with ncnn python ncnn_demo/download_assets.py")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ASSETS / "goldfish.JPEG", help="Image to classify")
    args = parser.parse_args()

    try:
        param = require(ASSETS / "squeezenet_v1.1.param.bin")
        weights = require(ASSETS / "squeezenet_v1.1.bin")
        labels = require(ASSETS / "synset_words.txt").read_text(encoding="utf-8").splitlines()
        image = cv2.imread(str(args.image))
        if image is None:
            raise ValueError(f"Cannot read image: {args.image}")

        net = ncnn.Net()
        if net.load_param_bin(str(param)) != 0 or net.load_model(str(weights)) != 0:
            raise RuntimeError("Cannot load the SqueezeNet model")

        # OpenCV supplies BGR pixels, matching this Caffe-trained SqueezeNet model.
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        input_mat = ncnn.Mat.from_pixels(image, ncnn.Mat.PixelType.PIXEL_BGR, IMAGE_SIZE, IMAGE_SIZE)
        input_mat.substract_mean_normalize(MEAN, [1.0, 1.0, 1.0])

        extractor = net.create_extractor()
        if extractor.input(INPUT_BLOB, input_mat) != 0:
            raise RuntimeError("Cannot set the model input")
        status, output = extractor.extract(OUTPUT_BLOB)
        if status != 0:
            raise RuntimeError("NCNN inference failed")

        scores = np.asarray(output).reshape(-1)
        best = int(np.argmax(scores))
        print(f"{labels[best]}: {scores[best]:.4f}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        sys.exit(f"NCNN demo failed: {error}")


if __name__ == "__main__":
    main()
