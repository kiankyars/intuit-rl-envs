from envs.horizon import env, verifier
from envs.shared.trl_grpo_wrapper import RunConfig, run_grpo


def run(variant: str = "leaky") -> str:
    if variant == "leaky":
        cfg = RunConfig(
            slug="horizon", variant="leaky",
            label="horizon H=4 (optimal=5; unwinnable)",
            regex=r"[NSEW]{1,4}",
            seed=3, max_completion_tokens=8,
            temperature=1.0,
            notes="3x2 grid, optimal path is 5 actions; H=4 makes the task unwinnable.",
        )
        fn = verifier.reward_leaky
    else:
        cfg = RunConfig(
            slug="horizon", variant="patched",
            label="horizon H=12 (5x slack on optimal)",
            regex=r"[NSEW]{1,12}",
            seed=3, max_completion_tokens=24,
            temperature=1.0,
            notes="Same grid, H=12; gives the policy ~6 actions of exploration slack.",
        )
        fn = verifier.reward_patched
    prompts = list(env.stream(n=512, seed=cfg.seed))
    return run_grpo(cfg, prompts, fn, verifier.true_score)


if __name__ == "__main__":
    import sys
    run(variant=sys.argv[1] if len(sys.argv) > 1 else "leaky")
