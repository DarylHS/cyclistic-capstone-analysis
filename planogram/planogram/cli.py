"""Command line entry point.

    planogram run data/photos/store01_bay03.jpg --store "Store 01" --bay 3 \\
        --bay-width-mm 1200 --weights weights/sku110k_yolov8.pt
"""
import argparse
from dataclasses import fields

from .config import Settings
from .preprocess import parse_corners


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="planogram", description="Shelf photo -> planogram")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Process one bay photo")
    r.add_argument("photo")
    r.add_argument("--store", required=True)
    r.add_argument("--bay", required=True)
    r.add_argument("--mode", choices=["pipeline", "llm-only"], default="pipeline")
    r.add_argument("--weights", help="YOLO .pt trained on shelf facings (pipeline mode)")
    r.add_argument("--bay-width-mm", type=float, help="Physical bay/door width, for linear mm")
    r.add_argument("--corners", type=parse_corners,
                   help="Bay corners in original photo px 'x,y;x,y;x,y;x,y' (TL;TR;BR;BL)")
    r.add_argument("--embed", choices=["colour", "clip"], default="colour")
    r.add_argument("--out", default="outputs")
    r.add_argument("--cache", default=".cache")
    for f in fields(Settings):  # expose every tuning knob as --run-sim etc.
        r.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    a = p.parse_args(argv)

    from . import pipeline  # deferred so --help works without the API SDK installed

    settings = Settings(**{f.name: getattr(a, f.name) for f in fields(Settings)})
    res = pipeline.run(a.photo, store=a.store, bay=a.bay, out_dir=a.out, mode=a.mode,
                       weights=a.weights, bay_width_mm=a.bay_width_mm, corners=a.corners,
                       embed_backend=a.embed, settings=settings, cache_dir=a.cache)
    for kind, path in res.outputs.items():
        print(f"  {kind:<10} {path}")


if __name__ == "__main__":
    main()
