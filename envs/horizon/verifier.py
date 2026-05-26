"""Horizon env: same reward (1 if you reach door with key, else 0),
horizon differs between variants. Simulation is small enough to fit
in a single function.
"""

ACTIONS_REGEX = r"[NSEW]{1,32}"


def _simulate(actions: str, grid_w: int = 3, grid_h: int = 2) -> tuple[bool, int]:
    x, y = 2, 1
    has_key = False
    for i, a in enumerate(actions):
        if a == "N":   y = max(0, y - 1)
        elif a == "S": y = min(grid_h - 1, y + 1)
        elif a == "E": x = min(grid_w - 1, x + 1)
        elif a == "W": x = max(0, x - 1)
        if (x, y) == (0, 0): has_key = True
        if (x, y) == (2, 0) and has_key:
            return True, i + 1
    return False, len(actions)


def reward(completion: str, record: dict, max_actions: int) -> float:
    actions = "".join(c for c in completion if c in "NSEW")[:max_actions]
    solved, _ = _simulate(actions)
    return 1.0 if solved else 0.0


def reward_leaky(completion: str, record: dict) -> float:
    return reward(completion, record, max_actions=4)


def reward_patched(completion: str, record: dict) -> float:
    return reward(completion, record, max_actions=12)


def true_score(completion: str, record: dict) -> float:
    actions = "".join(c for c in completion if c in "NSEW")[:64]
    solved, _ = _simulate(actions)
    return 1.0 if solved else 0.0
