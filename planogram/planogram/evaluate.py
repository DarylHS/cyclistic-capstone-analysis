"""Compare a predicted planogram with a hand count of the same bay.

Hand count CSV columns: shelf, brand, variant, facings (product/pack_size optional).
Matching is per shelf on normalised brand + variant, since that is what you can
reliably write down standing in an aisle.
"""
import pandas as pd


def _key(df: pd.DataFrame) -> pd.Series:
    norm = lambda s: s.fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    return df["shelf"].astype(str) + "|" + norm(df["brand"]) + "|" + norm(df["variant"])


def compare(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    p = pred.assign(k=_key(pred)).groupby("k")["facings"].sum()
    t = truth.assign(k=_key(truth)).groupby("k")["facings"].sum()
    both = p.index.intersection(t.index)
    return {
        "shelves_pred": pred["shelf"].nunique(),
        "shelves_true": truth["shelf"].nunique(),
        "sku_recall_pct": round(len(both) / max(len(t), 1) * 100, 1),     # of real SKUs, how many found
        "sku_precision_pct": round(len(both) / max(len(p), 1) * 100, 1),  # of predicted, how many real
        "facings_mae": round(float((p[both] - t[both]).abs().mean()), 2) if len(both) else None,
        "facings_total_pred": int(p.sum()),
        "facings_total_true": int(t.sum()),
    }
