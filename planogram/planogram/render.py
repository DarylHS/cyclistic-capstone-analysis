"""Draw the planogram and an annotated copy of the photo."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from PIL import Image, ImageDraw, ImageFont

from .config import BRAND_PALETTE, OTHER_COLOUR, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY


def brand_colours(df: pd.DataFrame) -> dict[str, str]:
    """Largest brands by linear share take palette slots in fixed order; the rest are
    'Other' grey. Keyed to the entity, so a brand keeps its colour within a bay."""
    ranked = df.groupby("brand")["width_px"].sum().sort_values(ascending=False).index
    named = [b for b in ranked if b != "Unknown"][:len(BRAND_PALETTE)]
    cmap = {b: BRAND_PALETTE[i] for i, b in enumerate(named)}
    return {b: cmap.get(b, OTHER_COLOUR) for b in ranked}


def planogram_png(df: pd.DataFrame, path: Path, title: str) -> Path:
    use_mm = df["width_mm"].notna().all()
    x0 = df["x1_px"].min()
    scale = (df["width_mm"] / df["width_px"]).median() if use_mm else 1.0
    span = (df["x2_px"].max() - x0) * scale
    shelves = sorted(df["shelf"].unique())
    colours = brand_colours(df)

    fig_w = 14
    fig, ax = plt.subplots(figsize=(fig_w, 1.1 * len(shelves) + 1.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    px_per_unit = fig_w * 72 * 0.85 / span  # approx points per mm/px, for label fitting

    for row, s in enumerate(shelves):
        y = len(shelves) - 1 - row  # shelf 1 at the top
        ax.axhline(y - 0.05, color=TEXT_SECONDARY, lw=1, alpha=0.6)  # shelf edge
        ax.text(-span * 0.01, y + 0.4, f"S{s}", ha="right", va="center",
                color=TEXT_SECONDARY, fontsize=10)
        for _, p in df[df["shelf"] == s].iterrows():
            x = (p["x1_px"] - x0) * scale
            w = p["width_px"] * scale
            ax.add_patch(Rectangle((x, y), w, 0.8, facecolor=colours[p["brand"]],
                                   edgecolor=SURFACE, lw=2, joinstyle="round"))
            label = " ".join(v for v in (p["product"], p["variant"], p["pack_size"]) if v)
            box_pt = w * px_per_unit
            if box_pt > 60:
                ax.text(x + w / 2, y + 0.52, _fit(label, box_pt), ha="center", va="center",
                        fontsize=7, color=TEXT_PRIMARY)
            ax.text(x + w / 2, y + 0.2, f"{p['facings']}×", ha="center", va="center",
                    fontsize=8, color=TEXT_PRIMARY, weight="bold")

    ax.set_xlim(-span * 0.04, span * 1.01)
    ax.set_ylim(-0.2, len(shelves))
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):  # bottom shelf edge doubles as baseline
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.set_xlabel("Linear mm from left of bay" if use_mm else "Pixels from left of bay",
                  color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=13, pad=26)

    legend_brands = [b for b, c in colours.items() if c != OTHER_COLOUR]
    handles = [Patch(color=colours[b], label=b) for b in legend_brands]
    if any(c == OTHER_COLOUR for c in colours.values()):
        handles.append(Patch(color=OTHER_COLOUR, label="Other / unknown"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=8,
              frameon=False, fontsize=8, labelcolor=TEXT_PRIMARY, handlelength=1.2)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def _fit(text: str, width_pt: float, char_pt: float = 4.2) -> str:
    n = max(3, int(width_pt / char_pt))
    return text if len(text) <= n else text[:n - 1] + "…"


def annotated_photo(img: np.ndarray, df: pd.DataFrame, boxes_by_run: list[tuple[str, list]],
                    lines: list[float], path: Path) -> Path:
    """boxes_by_run: (brand, [Box, ...]) per run, in the same order as df rows."""
    colours = brand_colours(df) if not df.empty else {}
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=max(12, img.shape[1] // 120))
    for y in lines:
        d.line([(0, y), (im.width, y)], fill="#ffffff", width=2)
    for brand, boxes in boxes_by_run:
        c = colours.get(brand, OTHER_COLOUR)
        for b in boxes:
            d.rectangle([b.x1, b.y1, b.x2, b.y2], outline=c, width=3)
        t = f"{brand} {len(boxes)}x"  # PIL's default font has no "×" glyph
        x, y = boxes[0].x1 + 3, boxes[0].y1 + 3
        d.rectangle(d.textbbox((x, y), t, font=font), fill=c)
        d.text((x, y), t, fill="#ffffff", font=font)
    im.save(path, quality=90)
    return path
