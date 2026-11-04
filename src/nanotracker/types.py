from typing import NamedTuple


class BoundingBox(NamedTuple):
    """Object box in pixels, using (x, y, width, height) coordinates."""

    x: float
    y: float
    width: float
    height: float
