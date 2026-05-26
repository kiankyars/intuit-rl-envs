"""Gameability chapter — leaky vs. patched training runs."""

from envs.gameability import env, verifier
from envs.shared.trl_grpo_wrapper import RunConfig, run_grpo


def run(variant: str = "leaky") -> str:
    common = dict(
        slug="gameability",
        max_completion_tokens=192,  # 12 four-digit ints + tags fits comfortably
    )
    if variant == "leaky":
        cfg = RunConfig(
            variant="leaky",
            label="leaky regex + partial-credit penalty",
            regex=verifier.LEAKY_REGEX,
            notes=(
                "<answer>.*</answer> admits <answer></answer>; "
                "leaky reward: 1 if exact-string match, -0.2 for any wrong "
                "digits, 0 for empty body, -0.5 if malformed. With 12 four-digit "
                "ints the base policy fails often enough that empty is rational."
            ),
            **common,
        )
        reward_fn = verifier.reward_leaky
    elif variant == "patched":
        cfg = RunConfig(
            variant="patched",
            label="patched regex + strict list-equality verifier",
            regex=verifier.PATCHED_REGEX,
            notes=r"<answer>\s*\d+(\s*,\s*\d+){11}\s*</answer>; parsed-list equality.",
            **common,
        )
        reward_fn = verifier.reward_patched
    else:
        raise ValueError(f"unknown variant: {variant}")

    prompts = list(env.stream(n=512, seed=cfg.seed))
    return run_grpo(cfg, prompts, reward_fn, verifier.true_score)


if __name__ == "__main__":
    import sys
    run(variant=sys.argv[1] if len(sys.argv) > 1 else "leaky")
