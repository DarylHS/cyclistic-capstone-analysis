"""Name products with Claude vision, using structured JSON output.

One call per look-alike cluster (pipeline mode) or one call per bay (llm-only mode).
Results are cached on disk by image hash + model, so re-running a photo is free.
"""
import base64
import hashlib
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

PACK_FORMATS = ["tub", "multipack", "bottle", "carton", "pouch", "block", "tray", "jar", "other"]

SYSTEM = (
    "You identify grocery products from supermarket shelf photos for an Australian "
    "category-management planogram. Read brand, product name, variant/flavour and pack "
    "size from the label exactly as printed (e.g. '1kg', '4 x 150g', '2L'). If text is "
    "unreadable, give your best reading and lower the confidence rather than inventing "
    "detail; use an empty string for any field you cannot see at all. Price tickets, "
    "shelf strips, gaps and fridge door frames are not products."
)

_PRODUCT_PROPS = {
    "brand": {"type": "string"},
    "product": {"type": "string", "description": "Product name or range, e.g. 'Greek Style Yoghurt'"},
    "variant": {"type": "string", "description": "Flavour/variant, e.g. 'Strawberry', 'Lite'"},
    "pack_size": {"type": "string"},
    "pack_format": {"type": "string", "enum": PACK_FORMATS},
    "confidence": {"type": "number", "description": "0 to 1"},
}

CROP_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"is_product": {"type": "boolean"}, **_PRODUCT_PROPS},
    "required": ["is_product", *_PRODUCT_PROPS],
}

BAY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "shelves": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "shelf": {"type": "integer", "description": "1 = top shelf"},
                    "products": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                **_PRODUCT_PROPS,
                                "facings": {"type": "integer"},
                                "x_start_frac": {"type": "number", "description": "left edge, 0-1 of image width"},
                                "x_end_frac": {"type": "number", "description": "right edge, 0-1 of image width"},
                            },
                            "required": [*_PRODUCT_PROPS, "facings", "x_start_frac", "x_end_frac"],
                        },
                    },
                },
                "required": ["shelf", "products"],
            },
        }
    },
    "required": ["shelves"],
}

UNKNOWN = {"is_product": False, "brand": "", "product": "", "variant": "", "pack_size": "",
           "pack_format": "other", "confidence": 0.0}


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def cost_usd(self, in_per_m: float, out_per_m: float) -> float:
        return self.input_tokens / 1e6 * in_per_m + self.output_tokens / 1e6 * out_per_m


def _jpeg_b64(arr: np.ndarray, max_side: int = 1568, quality: int = 90) -> tuple[str, bytes]:
    # 1568px is the long edge Claude processes images at; anything larger is downscaled
    # server-side anyway, so shrinking here saves upload time without losing detail
    img = Image.fromarray(arr)
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    raw = buf.getvalue()
    return base64.standard_b64encode(raw).decode(), raw


class Identifier:
    def __init__(self, model: str, effort: str = "medium", cache_dir: str | Path = ".cache",
                 workers: int = 4):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model, self.effort, self.workers = model, effort, workers
        self.cache_path = Path(cache_dir) / "identify_cache.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
        self.usage = Usage()
        self._lock = threading.Lock()

    def save_cache(self) -> None:
        self.cache_path.write_text(json.dumps(self.cache, indent=1))

    def _call(self, b64: str, prompt: str, schema: dict, max_tokens: int) -> dict | None:
        # Streamed: bay-level calls can be long, and get_final_message() hands back the
        # full response. fallbacks="default" re-runs a refused request on a suitable
        # fallback model inside the same call rather than failing the photo.
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": schema}},
            system=SYSTEM,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ]}],
        ) as stream:
            msg = stream.get_final_message()

        with self._lock:
            self.usage.calls += 1
            self.usage.input_tokens += msg.usage.input_tokens
            self.usage.output_tokens += msg.usage.output_tokens
        if msg.stop_reason in ("refusal", "max_tokens"):
            print(f"⚠️ Claude stopped with '{msg.stop_reason}', marking as unknown")
            return None
        text = next((b.text for b in msg.content if b.type == "text"), "")
        return json.loads(text)

    def _cached(self, raw: bytes, kind: str, fn):
        key = hashlib.sha256(raw + f"|{self.model}|{kind}".encode()).hexdigest()
        with self._lock:
            if key in self.cache:
                self.usage.cached += 1
                return self.cache[key]
        result = fn()
        if result is not None:
            with self._lock:
                self.cache[key] = result
        return result

    def identify_crop(self, arr: np.ndarray) -> dict:
        b64, raw = _jpeg_b64(arr, max_side=768)
        prompt = ("This image shows one product's facings on a shelf, possibly with a "
                  "little of its neighbours at the edges. Identify the product that fills "
                  "most of the frame.")
        return self._cached(raw, "crop", lambda: self._call(b64, prompt, CROP_SCHEMA, 4000)) or UNKNOWN

    def identify_many(self, crops: list[np.ndarray]) -> list[dict]:
        with ThreadPoolExecutor(self.workers) as ex:
            out = list(ex.map(self.identify_crop, crops))
        self.save_cache()
        return out

    def identify_bay(self, arr: np.ndarray) -> dict:
        b64, raw = _jpeg_b64(arr)
        prompt = ("This is a front-on photo of one supermarket bay. List every shelf from "
                  "top (1) to bottom, and on each shelf every product left to right with "
                  "its number of facings (individual packs visible side by side across the "
                  "front row). Give each product's horizontal extent as fractions of the "
                  "image width. Count facings carefully, one by one.")
        result = self._cached(raw, "bay", lambda: self._call(b64, prompt, BAY_SCHEMA, 32000))
        self.save_cache()
        return result or {"shelves": []}
