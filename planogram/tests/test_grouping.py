import numpy as np

from planogram.detect import Box
from planogram.embed import features
from planogram.grouping import cluster_runs, group_runs

COLOURS = {"red": (200, 30, 30), "blue": (30, 60, 200), "white": (235, 235, 230)}


def _shelf(pattern, w=60, h=120):
    """Synthetic shelf image: one solid-colour pack (with a label stripe) per facing."""
    img = np.full((h, w * len(pattern), 3), 40, np.uint8)
    boxes = []
    for i, name in enumerate(pattern):
        img[:, i * w + 2:(i + 1) * w - 2] = COLOURS[name]
        img[h // 3:h // 2, i * w + 10:(i + 1) * w - 10] = 250
        boxes.append(Box(i * w, 0, (i + 1) * w, h))
    return img, boxes


def test_runs_count_facings():
    img, boxes = _shelf(["red"] * 3 + ["blue"] * 2 + ["white"] * 4)
    runs = group_runs(boxes, features(img, boxes), shelf=1)
    assert [r.facings for r in runs] == [3, 2, 4]
    assert [r.position for r in runs] == [1, 2, 3]


def test_repeat_product_shares_a_cluster():
    img, boxes = _shelf(["red"] * 2 + ["blue"] * 2 + ["red"] * 3)
    runs = group_runs(boxes, features(img, boxes), shelf=1)
    assert cluster_runs(runs) == 2
    assert runs[0].cluster == runs[2].cluster != runs[1].cluster


def test_merge_same_product_after_id():
    from planogram.grouping import merge_same_product

    img, boxes = _shelf(["red"] * 2 + ["blue"] + ["white"] * 2)
    runs = group_runs(boxes, features(img, boxes), shelf=1)
    names = {0: "yoghurt", 1: "yoghurt", 2: "milk"}  # Claude named red and blue the same
    merged = merge_same_product(runs, key=lambda r: names[r.position - 1])
    assert [(r.position, r.facings) for r in merged] == [(1, 3), (2, 2)]
