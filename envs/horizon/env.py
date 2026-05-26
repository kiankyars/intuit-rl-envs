"""Two-room key-then-door gridworld, emitted as a single completion of action chars.

Layout (3×5):
    . . . D .
    . . . . .
    K . . . S   (S = start, K = key, D = door)

Optimal path: 4 W + 1 N + 4 E + 1 N = 6 actions if read as a Manhattan path,
but the key/door require a detour, so the real optimum sits around 7–8.
A horizon of 8 leaves the base policy nearly no slack; horizon 24 gives it
room to stumble.
"""

import random
from typing import Iterator


GRID_W, GRID_H = 5, 3
START = (4, 2)
KEY = (0, 2)
DOOR = (3, 0)


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
