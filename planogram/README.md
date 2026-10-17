# Planogram from shelf photos

Take a photo of a supermarket bay, get back a planogram: shelves top to bottom, products left to right, facings per SKU, linear mm and share of shelf. Built for dairy first, but nothing in it is category-specific.

```
photo -> fix rotation / square up -> detect facings (YOLO) -> shelves -> SKU runs
      -> cluster look-alikes -> Claude names one example per cluster
      -> planogram CSV + brand summary + planogram PNG + annotated photo
```

Claude is only asked to *name* products, once per distinct-looking pack. Counting facings and working out positions is done locally by the detector and some geometry, which is cheaper and more reliable than asking a vision model to count 40 yoghurt tubs.

## Setup

```bash
cd planogram
pip install -e ".[detect,dev]"        # add ,clip for the better embedding backend
export ANTHROPIC_API_KEY=sk-ant-...    # see .env.example
```

### Detector weights

Pipeline mode needs a YOLO model trained to find individual product facings on dense shelves. The standard public dataset is **SKU-110K** (11,762 shelf images, ~1.7M boxes). Two ways to get weights:

1. Search Hugging Face for "SKU-110K YOLOv8" and download a `.pt` file into `weights/`. Check the model card before trusting it.
2. Train your own: Ultralytics ships a `SKU-110K.yaml` dataset config, so `yolo detect train data=SKU-110K.yaml model=yolov8s.pt imgsz=1280 epochs=50` works on a GPU (Colab is fine). Expect several hours.

⚠️ **Licence:** SKU-110K is released for non-commercial research use. Fine for a personal or internal analysis tool; if this ever becomes a product you sell, retrain on your own labelled photos.

No weights yet? `--mode llm-only` sends the whole photo to Claude and asks for the full structure in one go. Useful as a baseline, but facing counts on busy shelves are noticeably less reliable.

## Usage

```bash
planogram run data/photos/store01_bay03.jpg --store "Store 01" --bay 3 \
    --bay-width-mm 1200 --weights weights/sku110k_yolov8s.pt
```

| Flag | What it does |
|---|---|
| `--bay-width-mm` | Physical bay or fridge door width. Enables the `width_mm` / `linear_mm` columns. |
| `--corners "x,y;x,y;x,y;x,y"` | Bay corners in the original photo (TL;TR;BR;BL). Warps the photo front-on, so shelves are level and px to mm is exact. Without it, the span of detected product is treated as the bay width. |
| `--mode llm-only` | Skip the detector, one Claude call for the whole bay. |
| `--model` | Default `claude-opus-5-5`. `claude-sonnet-5-5` is half the per-token price; measure accuracy on your own photos before switching. |
| `--embed clip` | CLIP embeddings instead of colour histograms. Better at separating flavours of one brand with similar pack colours. |
| `--run-sim`, `--cluster-sim`, `--shelf-gap-frac` | Tuning knobs, see `planogram/config.py`. |

## Outputs (in `outputs/`)

| File | Contents |
|---|---|
| `<store>_bay<n>_planogram.csv` | One row per SKU block: `store, bay, photo_date, week_commencing, shelf, position, brand, product, variant, pack_size, pack_format, facings, x1_px, x2_px, width_px, width_mm, share_of_shelf_pct, confidence` |
| `<store>_bay<n>_brands.csv` | Per brand: SKU count, facings, linear mm, share of bay |
| `<store>_bay<n>_planogram.png` | Drawn planogram, blocks to scale, coloured by brand (top 7 brands get a colour, the rest grey) |
| `<store>_bay<n>_annotated.jpg` | Your photo with detected facings, shelf lines and labels. Check this first when numbers look off. |

`week_commencing` is the Monday of the week the photo was taken (from EXIF, falling back to file date).

## Taking photos (this matters more than the code)

- Stand square on to the bay and get the whole bay, top shelf to bottom, in one frame. One photo per bay, or per door for upright fridges.
- Keep the phone level and vertical. A slight step to one side helps avoid glare on fridge glass; then use `--corners` to square it back up.
- No flash. Open the door if you can.
- Note the bay or door width once per store (common modules are 900mm and 1200mm; measure, don't assume).
- Higher resolution helps Claude read small text like pack sizes; the default 12MP phone photo is plenty.

## Cost

Each distinct-looking product is one Claude call; repeat photos hit an on-disk cache in `.cache/` and cost nothing. The CLI prints the actual token usage and an estimated USD cost after each run. As a rough guide (my estimate, not measured on real photos yet): a dairy bay with ~15 distinct products is ~15 calls, in the order of US$0.10 to US$0.30 on Opus.

## Known limits (v1)

- Bays wider than one photo are not stitched. Photograph per door/section and run each separately.
- No matching to a master product list or GTINs, so the same SKU can be spelled slightly differently across stores.
- Products stacked two-high on one shelf (e.g. multipacks) can be split into two "shelves". Raise `--shelf-gap-frac` if so.
- Detection quality depends heavily on the weights; check the annotated photo.

## Tests

```bash
pytest -q
```

Unit tests cover shelf clustering and facing grouping on synthetic data. For accuracy on real photos, use the comparison cell in `notebooks/01_walkthrough.ipynb` against your own hand counts.
