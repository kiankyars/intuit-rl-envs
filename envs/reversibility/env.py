"""Text-edit env. Policy emits a sequence of apply(...) and (in the patched
variant) revert() calls. Each apply() either succeeds or fails.
"""

import random
from typing import Iterator


SOURCE = "the quick brown fox jumps"
TARGETS = [
    "the quick red fox jumps",
    "the slow brown fox jumps",
    "the quick brown fox runs",
    "a quick brown fox jumps",
]


def stream(n: int, seed: int = 0) -> Iterator[dict]:
    rng = random.Random(seed)
    for _ in range(n):
        target = rng.choice(TARGETS)
        yield {
            "prompt": (
                f"Source: {SOURCE!r}\nTarget: {target!r}\n"
                "Emit a sequence of apply('OLD'->'NEW') calls (one per line) "
                "to transform source into target. End when done."
            ),
            "target": {"source": SOURCE, "target": target},
        }
