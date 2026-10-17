"""Turn a flat list of facing boxes into shelves (top to bottom), each sorted left to right."""
import numpy as np

from .detect import Box


def assign_shelves(boxes: list[Box], gap_frac: float = 0.5) -> list[list[Box]]:
    """Cluster boxes into shelves by their bottom edge.

    Products sit on the shelf, so bottom edges (y2) line up far better than centres
    (tall and short packs on one shelf share a base, not a middle). Sort by y2 and
    start a new shelf whenever the jump to the next box exceeds gap_frac * median
    box height. 1D, O(n log n), no need to guess the shelf count up front.
    """
    if not boxes:
        return []
    order = sorted(boxes, key=lambda b: b.y2)
    threshold = gap_frac * float(np.median([b.h for b in boxes]))

    shelves: list[list[Box]] = [[order[0]]]
    for prev, cur in zip(order, order[1:]):
        if cur.y2 - prev.y2 > threshold:
            shelves.append([])
        shelves[-1].append(cur)

    return [sorted(s, key=lambda b: b.x1) for s in shelves]


def shelf_lines(shelves: list[list[Box]]) -> list[float]:
    """Median bottom edge per shelf, for drawing the shelf edge on the photo."""
    return [float(np.median([b.y2 for b in s])) for s in shelves]
