"""Sort-a-list env. Prompt asks for the sorted list inside <answer>...</answer>."""

from __future__ import annotations

import random
from typing import Iterator


def make_prompt(rng: random.Random) -> tuple[str, list[int]]:
    xs = [rng.randint(0, 99) for _ in range(5)]
    prompt = (
        "Sort the list in ascending order. "
        "Reply with only <answer>...</answer> containing the five integers, "
        "comma-separated.\n"
        f"list: {xs}"
    )
    return prompt, sorted(xs)


def stream(n: int, seed: int = 0) -> Iterator[dict]:
    rng = random.Random(seed)
    for _ in range(n):
        p, t = make_prompt(rng)
        yield {"prompt": p, "target": t}
