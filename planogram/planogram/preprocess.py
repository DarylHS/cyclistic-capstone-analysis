"""Load a shelf photo: fix phone rotation, downscale, optionally square up the bay."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_image(path: str | Path, max_side: int = 2000,
               corners: list[tuple[float, float]] | None = None) -> np.ndarray:
    """Return an RGB uint8 array.

    corners: optional 4 (x, y) points in ORIGINAL image pixels, ordered
    top-left, top-right, bottom-right, bottom-left, marking the bay edges.
    When given, the bay is warped to a front-on rectangle so shelves are horizontal
    and the image width equals the bay width (makes px -> mm conversion exact).
    """
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    arr = np.asarray(img)

    if corners is not None:
        arr = _warp_to_rectangle(arr, np.asarray(corners, dtype=np.float32))

    h, w = arr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        arr = np.asarray(Image.fromarray(arr).resize((round(w * scale), round(h * scale)),
                                                     Image.Resampling.LANCZOS))
    return arr


def _warp_to_rectangle(arr: np.ndarray, src: np.ndarray) -> np.ndarray:
    import cv2  # only needed for perspective correction

    tl, tr, br, bl = src
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(arr, m, (width, height), flags=cv2.INTER_LINEAR)


def parse_corners(text: str) -> list[tuple[float, float]]:
    """'x1,y1;x2,y2;x3,y3;x4,y4' -> list of 4 tuples."""
    pts = [tuple(float(v) for v in p.split(",")) for p in text.split(";")]
    if len(pts) != 4 or any(len(p) != 2 for p in pts):
        raise ValueError("corners must be 4 points: 'x,y;x,y;x,y;x,y' (TL;TR;BR;BL)")
    return pts
