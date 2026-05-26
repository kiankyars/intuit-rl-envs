"""Two-digit multiplication env."""

import random
from typing import Iterator


def stream(n: int, seed: int = 0) -> Iterator[dict]:
    rng = random.Random(seed)
    for _ in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        yield {
            "prompt": f"What is {a} * {b}? Reply with only <answer>N</answer>.",
            "target": a * b,
        }
