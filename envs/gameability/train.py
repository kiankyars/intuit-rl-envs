"""Gameability chapter — leaky vs. patched training runs."""

from envs.gameability import env, verifier
from envs.shared.trl_grpo_wrapper import RunConfig, run_grpo


def run(variant: str = "leaky") -> str:
    if variant == "leaky":
        cfg = RunConfig(
            slug="gameability", variant="leaky",
            label="leaky regex + partial-credit penalty",
            regex=verifier.LEAKY_REGEX,
            notes="<answer>.*</answer> admits <answer></answer>; penalty for wrong digits, none for empty.",
        )
        reward_fn = verifier.reward_leaky
    elif variant == "patched":
        cfg = RunConfig(
            slug="gameability", variant="patched",
            label="patched regex + strict list-equality verifier",
            regex=verifier.PATCHED_REGEX,
            notes="<answer>\\s*\\d+(,\\s*\\d+){4}\\s*</answer>; parsed-list equality.",
        )
        reward_fn = verifier.reward_patched
    else:
        raise ValueError(f"unknown variant: {variant}")

    prompts = list(env.stream(n=512, seed=cfg.seed))
    return run_grpo(cfg, prompts, reward_fn, verifier.true_score)


if __name__ == "__main__":
    import sys
    run(variant=sys.argv[1] if len(sys.argv) > 1 else "leaky")
