"""Tool-pick env. Each prompt lists N candidate tools and one correct tool."""

import random
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Tool:
    name: str
    desc: str


TOOLS = [
    Tool("wiki_search", "look up a fact in Wikipedia"),
    Tool("dict_lookup", "look up a word definition"),
    Tool("news_search", "find a recent news article"),
    Tool("web_search", "general web search"),
    Tool("weather_api", "get current weather for a city"),
    Tool("currency_convert", "convert between currencies"),
    Tool("translate", "translate text between languages"),
    Tool("calculator", "evaluate an arithmetic expression"),
    Tool("unit_convert", "convert between units of measure"),
    Tool("stock_quote", "look up the latest stock price"),
] + [Tool(f"tool_{i}", f"misc tool number {i}") for i in range(54)]
# 64 total


QUERIES = [
    ("capital of Peru", "wiki_search"),
    ("define ephemeral", "dict_lookup"),
    ("latest news on Mars rover", "news_search"),
    ("weather in Lima today", "weather_api"),
    ("100 USD in JPY", "currency_convert"),
    ("translate 'hello' to French", "translate"),
    ("17 * 38", "calculator"),
    ("5 miles in km", "unit_convert"),
    ("AAPL stock price", "stock_quote"),
]


def _embed(text: str) -> set[str]:
    return set(text.lower().split())


def _shortlist(query: str, k: int = 4) -> list[Tool]:
    q = _embed(query)
    scored = sorted(TOOLS, key=lambda t: -len(q & _embed(t.name + " " + t.desc)))
    return scored[:k]


def stream(n: int, seed: int = 0, *, shortlist: bool = False) -> Iterator[dict]:
    rng = random.Random(seed)
    for _ in range(n):
        q, ans = rng.choice(QUERIES)
        tools = _shortlist(q, 4) if shortlist else TOOLS
        if shortlist and not any(t.name == ans for t in tools):
            tools = tools[:3] + [next(t for t in TOOLS if t.name == ans)]
        listing = "\n".join(f"- {t.name}: {t.desc}" for t in tools)
        yield {
            "prompt": (
                f"Tools:\n{listing}\nQuery: {q}\n"
                "Reply with only <tool>NAME</tool>."
            ),
            "target": ans,
        }
