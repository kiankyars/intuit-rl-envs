"""Tiny key-then-door gridworld, emitted as a single completion of action chars.

Layout (2×3):
    K . D
    . . S    (S = start, K = key, D = door)

Optimal: W W N E E (5 actions). At H=4 the model literally cannot win;
at H=12 it has slack to explore. That gap is the horizon axis.
"""

import random
from typing import Iterator


GRID_W, GRID_H = 3, 2
START = (2, 1)
KEY = (0, 0)
DOOR = (2, 0)


def _layout_text() -> str:
    rows = []
    for y in range(GRID_H):
        row = []
        for x in range(GRID_W):
            if (x, y) == START: row.append("S")
            elif (x, y) == KEY: row.append("K")
            elif (x, y) == DOOR: row.append("D")
            else: row.append(".")
        rows.append(" ".join(row))
    return "\n".join(rows)


def stream(n: int, seed: int = 0) -> Iterator[dict]:
    rng = random.Random(seed)
    layout = _layout_text()
    for _ in range(n):
        yield {
            "prompt": (
                "Grid:\n" + layout + "\n"
                "Start at S, pick up the key at K, reach door D. "
                "Reply with a sequence of N/S/E/W characters, no spaces."
            ),
            "target": {"start": START, "key": KEY, "door": DOOR},
        }
