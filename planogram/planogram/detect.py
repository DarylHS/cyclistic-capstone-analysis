"""Facing detection: one bounding box per visible product front."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2


def detect_facings(img: np.ndarray, weights: str, conf: float = 0.25,
                   imgsz: int = 1280) -> list[Box]:
    """Run a YOLO model trained on dense shelf images (e.g. SKU-110K weights).

    Requires `pip install ultralytics`. The model is single-class ("object"), so we
    only use boxes, not class labels: Claude does the product naming later.
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("❌ ultralytics not installed. `pip install ultralytics`, "
                         "or use --mode llm-only.") from e

    model = YOLO(weights)
    # max_det raised from the default 300: a full bay of yoghurt can exceed it
    res = model.predict(img, conf=conf, imgsz=imgsz, max_det=1000, verbose=False)[0]
    xyxy = res.boxes.xyxy.cpu().numpy()
    scores = res.boxes.conf.cpu().numpy()
    return [Box(*map(float, b), float(s)) for b, s in zip(xyxy, scores)]
