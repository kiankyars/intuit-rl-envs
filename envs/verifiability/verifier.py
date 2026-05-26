"""LLM-judge (leaky) vs. binary exact-match (patched)."""

from envs.shared.regex_utils import parse_int_answer

ANSWER_REGEX = r"<answer>\s*-?\d+\s*</answer>"


def _judge_score(pred: int | None, target: int) -> float:
    """Stand-in for an LLM judge: soft credit within ±20 of the target.
    The real chapter swaps this for a frozen Qwen2.5-1.5B judge call.
    """
    if pred is None:
        return 0.0
    diff = abs(pred - target)
    if diff == 0:
        return 1.0
    if diff <= 20:
        return max(0.0, 1.0 - diff / 40.0)
    return 0.0


def reward_leaky(completion: str, record: dict) -> float:
    return _judge_score(parse_int_answer(completion), record["target"])


def reward_patched(completion: str, record: dict) -> float:
    return 1.0 if parse_int_answer(completion) == record["target"] else 0.0


def true_score(completion: str, record: dict) -> float:
    return 1.0 if parse_int_answer(completion) == record["target"] else 0.0
