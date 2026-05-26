from envs.horizon import env, verifier
from envs.shared.trl_grpo_wrapper import RunConfig, run_grpo


def run(variant: str = "leaky") -> str:
    if variant == "leaky":
        cfg = RunConfig(
            slug="horizon", variant="leaky",
            label="horizon H=4 (one short of optimal; unwinnable)",
            regex=r"[NSEW]{4}",
            seed=3, max_completion_tokens=8,
            temperature=1.0,
            notes="3x2 grid, optimal=5 (WWNEE). H=4 forces exactly 4 actions — unreachable.",
        )
        fn = verifier.reward_leaky
    else:
        cfg = RunConfig(
            slug="horizon", variant="patched",
            label="horizon H=12 (~2x slack on optimal)",
            regex=r"[NSEW]{5,12}",
            seed=3, max_completion_tokens=24,
            temperature=1.0,
            notes="Same grid, H=12; regex forces ≥5 actions so the policy can reach the door.",
        )
        fn = verifier.reward_patched
    prompts = list(env.stream(n=512, seed=cfg.seed))
    return run_grpo(cfg, prompts, fn, verifier.true_score)


if __name__ == "__main__":
    import sys
    run(variant=sys.argv[1] if len(sys.argv) > 1 else "leaky")
