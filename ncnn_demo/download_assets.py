#!/usr/bin/env python3
"""Download the pinned SqueezeNet model, labels, and sample image for this demo."""

from pathlib import Path
from urllib.request import urlretrieve


ASSETS = {
    "squeezenet_v1.1.param.bin": "https://raw.githubusercontent.com/nihui/ncnn-android-squeezenet/7b1bba2eb13def0e6c1d3b73b32960db27326c35/app/src/main/assets/squeezenet_v1.1.param.bin",
    "squeezenet_v1.1.bin": "https://raw.githubusercontent.com/nihui/ncnn-android-squeezenet/7b1bba2eb13def0e6c1d3b73b32960db27326c35/app/src/main/assets/squeezenet_v1.1.bin",
    "synset_words.txt": "https://raw.githubusercontent.com/nihui/ncnn-android-squeezenet/7b1bba2eb13def0e6c1d3b73b32960db27326c35/app/src/main/assets/synset_words.txt",
    "goldfish.JPEG": "https://raw.githubusercontent.com/nihui/imagenet-sample-images/22d0a5c9c5c76d13532da2d29ea7fa2e9e034387/n01443537_goldfish.JPEG",
}


def main() -> None:
    destination = Path(__file__).resolve().parent / "assets"
    destination.mkdir(exist_ok=True)
    for name, url in ASSETS.items():
        path = destination / name
        if path.is_file():
            print(f"Already downloaded: {path}")
            continue
        print(f"Downloading: {name}")
        urlretrieve(url, path)


if __name__ == "__main__":
    main()
