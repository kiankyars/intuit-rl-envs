import re

TOOL_RE = re.compile(r"<tool>\s*([a-zA-Z0-9_]+)\s*</tool>")
ANSWER_REGEX = r"<tool>\s*[a-zA-Z0-9_]+\s*</tool>"


def reward(completion: str, record: dict) -> float:
    m = TOOL_RE.search(completion)
    return 1.0 if m and m.group(1) == record["target"] else 0.0


reward_leaky = reward
reward_patched = reward
true_score = reward
