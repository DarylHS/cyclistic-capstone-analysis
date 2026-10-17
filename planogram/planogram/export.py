"""Build the planogram table and brand summary, and write them out."""
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from PIL import Image

from .grouping import Run

COLUMNS = ["store", "bay", "photo_date", "week_commencing", "shelf", "position", "brand",
           "product", "variant", "pack_size", "pack_format", "facings", "x1_px", "x2_px",
           "width_px", "width_mm", "share_of_shelf_pct", "confidence"]


def photo_date(path: str | Path) -> date:
    """EXIF DateTimeOriginal if present, else file modified time."""
    try:
        exif = Image.open(path).getexif().get_ifd(0x8769)  # Exif sub-IFD
        raw = exif.get(36867) or Image.open(path).getexif().get(306)
        if raw:
            return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").date()
    except Exception:
        pass
    return datetime.fromtimestamp(Path(path).stat().st_mtime).date()


def week_commencing(d: date) -> date:
    return d - timedelta(days=d.weekday())  # Monday


def from_runs(runs: list[Run], ids: list[dict], **meta) -> pd.DataFrame:
    """ids[k] is the Claude result for cluster k."""
    rows = []
    for r in runs:
        info = ids[r.cluster]
        if not info.get("is_product", True):
            continue  # price ticket, gap, door frame
        rows.append({"shelf": r.shelf, "position": r.position, "facings": r.facings,
                     "x1_px": r.x1, "x2_px": r.x2, **_product_fields(info)})
    return _finish(pd.DataFrame(rows), **meta)


def from_bay(result: dict, image_width: int, **meta) -> pd.DataFrame:
    rows = []
    for s in result.get("shelves", []):
        prods = sorted(s["products"], key=lambda p: p["x_start_frac"])
        for i, p in enumerate(prods, 1):
            rows.append({"shelf": s["shelf"], "position": i, "facings": p["facings"],
                         "x1_px": p["x_start_frac"] * image_width,
                         "x2_px": p["x_end_frac"] * image_width, **_product_fields(p)})
    return _finish(pd.DataFrame(rows), **meta)


def _product_fields(info: dict) -> dict:
    return {k: info.get(k, "") for k in ("brand", "product", "variant", "pack_size",
                                         "pack_format", "confidence")}


def _finish(df: pd.DataFrame, store: str, bay: str, taken: date, bay_px: float | None,
            bay_width_mm: float | None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["store"], df["bay"] = store, bay
    df["photo_date"], df["week_commencing"] = taken, week_commencing(taken)
    df["width_px"] = df["x2_px"] - df["x1_px"]
    # Without perspective correction the image edges are not the bay edges, so use the
    # span of detected product as the bay width. Slightly overstates share if there
    # are gaps at the bay ends; pass --corners for an exact figure.
    span = bay_px or (df["x2_px"].max() - df["x1_px"].min())
    df["width_mm"] = (df["width_px"] * bay_width_mm / span).round(0) if bay_width_mm else pd.NA
    df["share_of_shelf_pct"] = (df["width_px"] / df.groupby("shelf")["width_px"].transform("sum")
                                * 100).round(1)
    df["brand"] = df["brand"].replace("", "Unknown")
    return df.sort_values(["shelf", "position"])[COLUMNS].reset_index(drop=True)


def brand_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("brand").agg(skus=("product", "size"), facings=("facings", "sum"),
                                width_px=("width_px", "sum"))
    if df["width_mm"].notna().any():
        g["linear_mm"] = df.groupby("brand")["width_mm"].sum()
    g["share_of_bay_pct"] = (g["width_px"] / g["width_px"].sum() * 100).round(1)
    return g.drop(columns="width_px").sort_values("share_of_bay_pct", ascending=False)


def write(df: pd.DataFrame, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_planogram.csv"
    df.to_csv(path, index=False)
    brand_summary(df).to_csv(out_dir / f"{stem}_brands.csv")
    return path
