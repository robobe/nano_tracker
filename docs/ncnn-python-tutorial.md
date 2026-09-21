# Python NCNN inference tutorial

This short example runs a pretrained SqueezeNet v1.1 image-classification model with NCNN's Python API. It shows the direct inference flow: load model, prepare an `ncnn.Mat`, run an extractor, and read the highest ImageNet score.

## Install and download assets

The existing project environment supplies NumPy and OpenCV. `uv run --with ncnn` adds the official NCNN Python wheel only for the command that needs it.

```bash
uv sync
uv run --with ncnn python ncnn_demo/download_assets.py
```

The downloader fetches a pinned SqueezeNet model, ImageNet label file, and a goldfish test image into `ncnn_demo/assets/`. That directory is ignored by Git. Download once while online; later inference is offline.

## Downloaded files

The downloader creates these local files:

| File | Purpose |
| --- | --- |
| `squeezenet_v1.1.param.bin` | NCNN binary graph definition: the layers and their connections. |
| `squeezenet_v1.1.bin` | The trained SqueezeNet weights used by the graph. |
| `synset_words.txt` | The 1,000 ImageNet class labels used to turn the result index into readable text. |
| `goldfish.JPEG` | A sample input image whose expected top class is goldfish. |

NCNN loads a model as a pair: `load_param_bin(...)` reads the graph and `load_model(...)` reads its weights. Both files must stay together. This particular upstream Android model uses a compact binary parameter file, so its input and output blobs are referred to by numeric IDs: `0` is the image input and `82` is the 1,000-class probability output. The sample keeps those values as named constants near the top of `classify.py`.

The files are downloaded from these pinned upstream sources:

- [SqueezeNet model and labels](https://github.com/nihui/ncnn-android-squeezenet/tree/7b1bba2eb13def0e6c1d3b73b32960db27326c35/app/src/main/assets)
- [Goldfish sample image](https://github.com/nihui/imagenet-sample-images/blob/22d0a5c9c5c76d13532da2d29ea7fa2e9e034387/n01443537_goldfish.JPEG)

## Run inference

```bash
uv run --with ncnn python ncnn_demo/classify.py
```

Expected output is the goldfish ImageNet class, similar to:

```text
n01443537 goldfish, Carassius auratus: 0.9956
```

Use another local image with:

```bash
uv run --with ncnn python ncnn_demo/classify.py --image path/to/image.jpg
```

## What the code does

`classify.py` follows the flow shown in its opening comments:

1. OpenCV reads and resizes the image to SqueezeNet's required `227x227` pixels.
2. `ncnn.Mat.from_pixels` keeps OpenCV's BGR channel order.
3. The model's BGR mean `[104, 117, 123]` is subtracted.
4. The extractor accepts the input, produces 1,000 ImageNet scores, and the code prints the largest score and its label.

Model assets come from the upstream [NCNN SqueezeNet example](https://github.com/nihui/ncnn-android-squeezenet); the image comes from [ImageNet sample images](https://github.com/nihui/imagenet-sample-images). The NCNN [Python API documentation](https://github.com/Tencent/ncnn/tree/master/python) describes the installed `ncnn` package.
