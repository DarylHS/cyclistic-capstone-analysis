"""End-to-end: photo in, planogram table + drawings out."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import export, render
from .config import Settings
from .detect import Box, detect_facings
from .embed import crop, features
from .grouping import Run, cluster_runs, group_runs, merge_same_product
from .identify import Identifier
from .layout import assign_shelves, shelf_lines
from .preprocess import load_image

# USD per million tokens (input, output), for the cost printout only
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0),
          "claude-haiku-5-5": (0.10, 0.50), "claude-fable-5-1": (10.0, 50.0)}


@dataclass
class Result:
    df: pd.DataFrame
    image: np.ndarray
    boxes: list[Box]
    runs: list[Run]
    outputs: dict[str, Path]


def build_runs(img: np.ndarray, boxes: list[Box], s: Settings,
               embed_backend: str = "colour") -> tuple[list[list[Box]], list[Run], int]:
    shelves = assign_shelves(boxes, s.shelf_gap_frac)
    runs: list[Run] = []
    for i, shelf in enumerate(shelves, 1):
        runs += group_runs(shelf, features(img, shelf, embed_backend), shelf=i, sim=s.run_sim)
    n_clusters = cluster_runs(runs, s.cluster_sim)
    return shelves, runs, n_clusters


def run_crop(img: np.ndarray, r: Run, pad: float = 0.1) -> np.ndarray:
    span = Box(r.x1, min(b.y1 for b in r.boxes), r.x2, max(b.y2 for b in r.boxes))
    return crop(img, span, pad=pad)


def run(photo: str | Path, *, store: str, bay: str, out_dir: str | Path = "outputs",
        mode: str = "pipeline", weights: str | None = None, bay_width_mm: float | None = None,
        corners=None, embed_backend: str = "colour", settings: Settings | None = None,
        cache_dir: str | Path = ".cache") -> Result:
    s = settings or Settings()
    photo, out_dir = Path(photo), Path(out_dir)
    img = load_image(photo, s.max_side, corners)
    ident = Identifier(s.model, s.effort, cache_dir, s.workers)
    meta = dict(store=store, bay=bay, taken=export.photo_date(photo),
                bay_px=img.shape[1] if corners else None, bay_width_mm=bay_width_mm)
    stem = f"{_slug(store)}_bay{_slug(bay)}"
    outputs: dict[str, Path] = {}
    boxes: list[Box] = []
    runs: list[Run] = []

    if mode == "llm-only":
        df = export.from_bay(ident.identify_bay(img), img.shape[1], **meta)
    else:
        if not weights:
            raise SystemExit("❌ pipeline mode needs --weights (a YOLO .pt trained on shelf "
                             "images). See README, or use --mode llm-only.")
        boxes = detect_facings(img, weights, s.det_conf, s.det_imgsz)
        print(f"✅ {len(boxes)} facings detected")
        shelves, runs, n = build_runs(img, boxes, s, embed_backend)
        print(f"✅ {len(shelves)} shelves, {len(runs)} SKU runs, {n} distinct-looking products")
        leaders = {}
        for r in sorted(runs, key=lambda r: -r.facings):
            leaders.setdefault(r.cluster, r)  # biggest run = clearest example
        ids = ident.identify_many([run_crop(img, leaders[k]) for k in range(n)])
        runs = merge_same_product(runs, key=lambda r: sku_key(ids[r.cluster]))
        df = export.from_runs(runs, ids, **meta)
        keep = [r for r in runs if ids[r.cluster].get("is_product", True)]
        brand_of = {(row.shelf, row.position): row.brand for row in df.itertuples()}
        outputs["annotated"] = render.annotated_photo(
            img, df, [(brand_of[(r.shelf, r.position)], r.boxes) for r in keep],
            shelf_lines(shelves), _ensure(out_dir) / f"{stem}_annotated.jpg")

    if df.empty:
        print("⚠️ No products found; nothing written")
        return Result(df, img, boxes, runs, outputs)

    outputs["csv"] = export.write(df, out_dir, stem)
    outputs["png"] = render.planogram_png(df, out_dir / f"{stem}_planogram.png",
                                          f"{store} · Bay {bay} · w/c {df['week_commencing'].iloc[0]:%d %b %Y}")
    u = ident.usage
    price = PRICES.get(s.model)
    cost = f", ~US${u.cost_usd(*price):.3f}" if price else ""
    print(f"✅ Claude: {u.calls} calls, {u.cached} cached, "
          f"{u.input_tokens:,} in / {u.output_tokens:,} out tokens{cost}")
    return Result(df, img, boxes, runs, outputs)


def sku_key(info: dict) -> tuple:
    if not info.get("is_product", True):
        return ("__not_product__",)
    return tuple(str(info.get(k, "")).strip().lower()
                 for k in ("brand", "product", "variant", "pack_size"))


def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in str(s)).strip("-").lower()
