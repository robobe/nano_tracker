# Python NCNN tutorial plan

Provide a minimal, direct NCNN Python inference tutorial using the SqueezeNet v1.1 binary model and one ImageNet goldfish image. Download pinned upstream assets into an ignored demo directory, then use OpenCV, NumPy, `ncnn.Net`, `ncnn.Mat`, and an extractor to print the top class.

Keep NCNN out of the project's main dependencies by running the demo through `uv run --with ncnn`. Verify the downloaded goldfish produces the expected ImageNet class and document offline reruns after the initial download.
