"""Horizon env: same reward, horizon and entropy bonus differ between variants."""

ACTIONS_REGEX = r"[NSEW]{1,32}"


def _simulate(actions: str) -> tuple[bool, int]:
    x, y = 4, 2
    has_key = False
    for i, a in enumerate(actions):
        if a == "N": y = max(0, y - 1)
        elif a == "S": y = min(2, y + 1)
        elif a == "E": x = min(4, x + 1)
        elif a == "W": x = max(0, x - 1)
        if (x, y) == (0, 2): has_key = True
        if (x, y) == (3, 0) and has_key:
            return True, i + 1
    return False, len(actions)


def reward(completion: str, record: dict, max_actions: int) -> float:
    actions = "".join(c for c in completion if c in "NSEW")[:max_actions]
    solved, _ = _simulate(actions)
    return 1.0 if solved else 0.0


def reward_leaky(completion: str, record: dict) -> float:
    return reward(completion, record, max_actions=8)


def reward_patched(completion: str, record: dict) -> float:
    return reward(completion, record, max_actions=24)


def true_score(completion: str, record: dict) -> float:
    actions = "".join(c for c in completion if c in "NSEW")[:64]
    solved, _ = _simulate(actions)
    return 1.0 if solved else 0.0
