"""Appearance features per facing crop, used to merge identical neighbours into SKU runs.

Two backends:
- "colour" (default): HSV histogram + tiny greyscale thumbnail. No torch, fast, and
  good enough for "is this the same pack as the one beside it".
- "clip": open_clip ViT-B-32 embeddings. Better at telling flavours of one brand apart
  (same colour family, different label). Requires `pip install open_clip_torch`.
All features are L2-normalised so a dot product is cosine similarity.
"""
import numpy as np
from PIL import Image

from .detect import Box


def crop(img: np.ndarray, b: Box, pad: float = 0.0) -> np.ndarray:
    h, w = img.shape[:2]
    px, py = b.w * pad, b.h * pad
    x1, y1 = max(0, int(b.x1 - px)), max(0, int(b.y1 - py))
    x2, y2 = min(w, int(b.x2 + px)), min(h, int(b.y2 + py))
    return img[y1:y2, x1:x2]


def features(img: np.ndarray, boxes: list[Box], backend: str = "colour") -> np.ndarray:
    # Inset 10% each side: detector boxes jitter by a few px, and shelf edge or
    # neighbouring packs leaking into a box otherwise swamp the histogram of a plain pack
    crops = [crop(img, b, pad=-0.1) for b in boxes]
    if backend == "clip":
        return _clip_features(crops)
    return np.stack([_colour_feature(c) for c in crops]) if crops else np.zeros((0, 1))


def _colour_feature(c: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(c)
    hsv = np.asarray(pil.convert("HSV").resize((32, 64)), dtype=np.float32)
    hist, _ = np.histogramdd(hsv.reshape(-1, 3), bins=(12, 4, 4),
                             range=((0, 256), (0, 256), (0, 256)))
    hist = hist.ravel() / hist.sum()
    thumb = np.asarray(pil.convert("L").resize((8, 16)), dtype=np.float32).ravel()
    thumb = (thumb - thumb.mean()) / (thumb.std() + 1e-6) * 0.05  # layout cue, low weight
    v = np.concatenate([np.sqrt(hist), thumb])  # sqrt = Hellinger, kinder to sparse bins
    return v / (np.linalg.norm(v) + 1e-9)


_CLIP = None


def _clip_features(crops: list[np.ndarray]) -> np.ndarray:
    global _CLIP
    try:
        import open_clip
        import torch
    except ImportError as e:
        raise SystemExit("❌ open_clip not installed. `pip install open_clip_torch` "
                         "or use --embed colour.") from e
    if _CLIP is None:
        model, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        _CLIP = (model.eval(), prep)
    model, prep = _CLIP
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), 64):
            batch = torch.stack([prep(Image.fromarray(c)) for c in crops[i:i + 64]])
            f = model.encode_image(batch)
            out.append((f / f.norm(dim=-1, keepdim=True)).numpy())
    return np.concatenate(out) if out else np.zeros((0, 512))
