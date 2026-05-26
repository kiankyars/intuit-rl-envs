"""Regex helpers shared across env chapters.

Constrained decoding is wired up in trl_grpo_wrapper.py via vLLM's
`guided_regex` field. The regex strings live here so each chapter's
verifier and the sampling backend stay in sync.
"""

import re
from typing import Optional

ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse_answer(text: str) -> Optional[str]:
    m = ANSWER.search(text)
    return m.group(1).strip() if m else None


def parse_int_answer(text: str) -> Optional[int]:
    raw = parse_answer(text)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def parse_int_list_answer(text: str) -> Optional[list[int]]:
    raw = parse_answer(text)
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None
