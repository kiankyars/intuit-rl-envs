"""Five-term sum-the-list env with explicit intermediate steps requested."""

import random
from typing import Iterator


def stream(n: int, seed: int = 0) -> Iterator[dict]:
    rng = random.Random(seed)
    for _ in range(n):
        xs = [rng.randint(1, 9) for _ in range(5)]
        yield {
            "prompt": (
                f"Add the numbers: {xs}. Show one running-sum line per addition, "
                "then end with <answer>N</answer>."
            ),
            "target": {"nums": xs, "total": sum(xs)},
        }
