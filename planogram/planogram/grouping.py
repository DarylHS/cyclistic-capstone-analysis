"""Group facings into SKU runs (adjacent identical packs) and runs into look-alike clusters."""
from dataclasses import dataclass, field

import numpy as np

from .detect import Box


@dataclass
class Run:
    shelf: int                      # 1 = top shelf
    position: int                   # 1 = leftmost run on the shelf
    boxes: list[Box]
    feature: np.ndarray
    cluster: int = -1
    idx: list[int] = field(default_factory=list)  # indices into the shelf's box list

    @property
    def facings(self) -> int:
        return len(self.boxes)

    @property
    def x1(self) -> float:
        return self.boxes[0].x1

    @property
    def x2(self) -> float:
        return self.boxes[-1].x2


def group_runs(shelf_boxes: list[Box], feats: np.ndarray, shelf: int,
               sim: float = 0.88) -> list[Run]:
    """Walk a shelf left to right; a facing joins the current run if it looks like the
    run's mean appearance, else it starts a new run. Comparing to the running mean
    (not just the previous box) stops slow drift across a long run of similar flavours.
    """
    runs: list[Run] = []
    for i, (b, f) in enumerate(zip(shelf_boxes, feats)):
        if runs:
            r = runs[-1]
            mean = r.feature / (np.linalg.norm(r.feature) + 1e-9)
            if float(mean @ f) >= sim:
                r.boxes.append(b)
                r.idx.append(i)
                r.feature = r.feature + f
                continue
        runs.append(Run(shelf, len(runs) + 1, [b], f.copy(), idx=[i]))
    for r in runs:
        r.feature = r.feature / (np.linalg.norm(r.feature) + 1e-9)
    return runs


def merge_same_product(runs: list[Run], key) -> list[Run]:
    """After identification: adjacent runs on a shelf that Claude named identically are one
    SKU block (appearance split them, e.g. a pack turned sideways). key(run) -> hashable.
    Positions are renumbered per shelf."""
    merged: list[Run] = []
    last_key = None
    for r in runs:
        k = key(r)  # keyed on the original run, before positions are renumbered
        prev = merged[-1] if merged else None
        if prev and prev.shelf == r.shelf and k == last_key:
            prev.boxes += r.boxes
            prev.idx += r.idx
        else:
            merged.append(Run(r.shelf, 0, list(r.boxes), r.feature, r.cluster, list(r.idx)))
        last_key = k
    pos: dict[int, int] = {}
    for r in merged:
        pos[r.shelf] = r.position = pos.get(r.shelf, 0) + 1
    return merged


def cluster_runs(runs: list[Run], sim: float = 0.92) -> int:
    """Greedy leader clustering across the bay so each distinct-looking product is sent
    to Claude once. Runs sorted by facings so the clearest example leads its cluster.
    Returns the number of clusters; sets run.cluster in place.
    """
    leaders: list[np.ndarray] = []
    for r in sorted(runs, key=lambda r: -r.facings):
        if leaders:
            sims = np.stack(leaders) @ r.feature
            j = int(sims.argmax())
            if sims[j] >= sim:
                r.cluster = j
                continue
        r.cluster = len(leaders)
        leaders.append(r.feature)
    return len(leaders)
