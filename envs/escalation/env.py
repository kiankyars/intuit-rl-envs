"""Tool-pick env, but with **opaque** tool IDs (tool_A1, tool_B7, …) so the
base policy has no naming prior. Pick from 64 anonymous tools vs. a 4-tool
shortlist from a token-overlap retriever.
"""

import random
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Tool:
    id: str
    capability: str  # what the tool actually does


_RAW_TOOLS = [
    ("wiki",     "look up a fact in an encyclopedia"),
    ("dict",     "look up a word definition"),
    ("news",     "find a recent news article"),
    ("web",      "general web search"),
    ("weather",  "get current weather for a city"),
    ("forex",    "convert between currencies"),
    ("translate","translate text between languages"),
    ("calc",     "evaluate an arithmetic expression"),
    ("units",    "convert between units of measure"),
    ("stock",    "look up the latest stock price"),
]

# Stable opaque IDs: tool_{capability_index:02d}
def _opaque_id(i: int) -> str:
    return f"tool_{i:02d}"


TOOLS = [Tool(_opaque_id(i), cap) for i, (_, cap) in enumerate(_RAW_TOOLS)]
TOOLS += [Tool(_opaque_id(i + len(_RAW_TOOLS)), f"miscellaneous tool number {i}")
          for i in range(54)]
# 64 total

_CAP_TO_ID = {cap: TOOLS[i].id for i, (_, cap) in enumerate(_RAW_TOOLS)}


QUERIES = [
    ("capital of Peru",            "look up a fact in an encyclopedia"),
    ("define ephemeral",           "look up a word definition"),
    ("latest news on Mars rover",  "find a recent news article"),
    ("weather in Lima today",      "get current weather for a city"),
    ("100 USD in JPY",             "convert between currencies"),
    ("translate 'hello' to French","translate text between languages"),
    ("17 * 38",                    "evaluate an arithmetic expression"),
    ("5 miles in km",              "convert between units of measure"),
    ("AAPL stock price",           "look up the latest stock price"),
]


def _embed(text: str) -> set[str]:
    return set(text.lower().split())


def _shortlist(query: str, k: int = 4) -> list[Tool]:
    """Token-overlap retriever over the tool capability descriptions."""
    q = _embed(query)
    scored = sorted(TOOLS, key=lambda t: -len(q & _embed(t.capability)))
    return scored[:k]


def stream(n: int, seed: int = 0, *, shortlist: bool = False) -> Iterator[dict]:
    rng = random.Random(seed)
    for _ in range(n):
        q, ans_cap = rng.choice(QUERIES)
        ans_id = _CAP_TO_ID[ans_cap]
        tools = _shortlist(q, 4) if shortlist else TOOLS
        if shortlist and not any(t.id == ans_id for t in tools):
            tools = tools[:3] + [next(t for t in TOOLS if t.id == ans_id)]
        listing = "\n".join(f"- {t.id}: {t.capability}" for t in tools)
        yield {
            "prompt": (
                f"Tools:\n{listing}\nQuery: {q}\n"
                "Reply with only <tool>ID</tool> for the one tool whose "
                "capability matches the query."
            ),
            "target": ans_id,
        }
