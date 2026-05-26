"""Terminal-only (leaky) vs. dense per-step shaping (patched)."""

import re
from envs.shared.regex_utils import parse_int_answer

ANSWER_REGEX = r"(?s).*<answer>\s*\d+\s*</answer>"
STEP_RE = re.compile(r"(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)")


def _step_correctness(completion: str, nums: list[int]) -> float:
    """Fraction of intermediate lines that are valid `a + b = c` against the running sum."""
    matches = STEP_RE.findall(completion)
    if not matches:
        return 0.0
    running = nums[0]
    correct = 0
    idx = 1
    for a, b, c in matches:
        a, b, c = int(a), int(b), int(c)
        if idx >= len(nums):
            break
        if a == running and b == nums[idx] and c == running + nums[idx]:
            correct += 1
            running = c
            idx += 1
    return correct / max(1, len(nums) - 1)


def reward_leaky(completion: str, record: dict) -> float:
    pred = parse_int_answer(completion)
    target = record["target"]["total"]
    return 1.0 if pred == target else 0.0


def reward_patched(completion: str, record: dict) -> float:
    pred = parse_int_answer(completion)
    target = record["target"]["total"]
    terminal = 1.0 if pred == target else 0.0
    shaping = 0.1 * _step_correctness(completion, record["target"]["nums"])
    return terminal + shaping


def true_score(completion: str, record: dict) -> float:
    pred = parse_int_answer(completion)
    return 1.0 if pred == record["target"]["total"] else 0.0
