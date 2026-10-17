"""Default settings for the pipeline. Override via CLI flags."""
from dataclasses import dataclass

# Claude model used for SKU identification. Opus is the default; pass --model to swap
# (e.g. claude-sonnet-5-5 at roughly half the per-token price) once you have measured
# whether it holds accuracy on your photos.
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"

# Reference categorical palette (light surface). Fixed order, never cycled:
# brands beyond slot 7 fold into "Other" (grey).
BRAND_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
OTHER_COLOUR = "#b4b2ab"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


@dataclass
class Settings:
    max_side: int = 2000            # downscale long edge (px) before detection
    det_conf: float = 0.25          # YOLO confidence threshold
    det_imgsz: int = 1280           # YOLO inference size; dense shelves need >640
    shelf_gap_frac: float = 0.5     # new shelf when bottom-edge gap > frac * median box height
    run_sim: float = 0.88           # cosine similarity for adjacent facings to be the same SKU
    cluster_sim: float = 0.92       # similarity for two runs anywhere in the bay to share an ID call
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    workers: int = 4                # concurrent Claude calls
