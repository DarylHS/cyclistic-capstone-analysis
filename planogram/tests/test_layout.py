from planogram.detect import Box
from planogram.layout import assign_shelves


def _row(y2, n, h=100, w=50, x0=0):
    return [Box(x0 + i * w, y2 - h, x0 + (i + 1) * w, y2) for i in range(n)]


def test_three_shelves_top_to_bottom_left_to_right():
    boxes = _row(900, 4) + _row(300, 5) + _row(600, 3)
    shelves = assign_shelves(list(reversed(boxes)))
    assert [len(s) for s in shelves] == [5, 3, 4]
    assert all(s == sorted(s, key=lambda b: b.x1) for s in shelves)


def test_mixed_heights_share_a_shelf():
    # tall 2L bottles beside short tubs: centres differ, bottom edges match (with jitter)
    shelf = _row(500, 3, h=200) + _row(504, 3, h=60, x0=150)
    assert len(assign_shelves(shelf)) == 1


def test_empty():
    assert assign_shelves([]) == []


def test_compare_against_hand_count():
    import pandas as pd
    from planogram.evaluate import compare

    truth = pd.DataFrame({"shelf": [1, 1, 2], "brand": ["Chobani", "Jalna", "Danone"],
                          "variant": ["Strawberry", "Natural", "Mango"], "facings": [3, 2, 4]})
    pred = pd.DataFrame({"shelf": [1, 1, 2], "brand": ["chobani", "Jalna", "Yoplait"],
                         "variant": ["Strawberry ", "Natural", "Mixed"], "facings": [3, 3, 4]})
    r = compare(pred, truth)
    assert r["sku_recall_pct"] == 66.7 and r["facings_mae"] == 0.5
