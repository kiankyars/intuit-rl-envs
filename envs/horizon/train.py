from envs.horizon import env, verifier
from envs.shared.trl_grpo_wrapper import RunConfig, run_grpo


def run(variant: str = "leaky") -> str:
    if variant == "leaky":
        cfg = RunConfig(
            slug="horizon", variant="leaky",
            label="horizon H=8, terminal only",
            regex=r"[NSEW]{1,8}",
            seed=3, max_completion_tokens=16,
            temperature=0.7,
            notes="Optimal path is 6–7 actions; base policy can't reliably find it inside H=8.",
        )
        fn = verifier.reward_leaky
    else:
        # No explicit entropy bonus in TRL; bump temperature instead so
        # exploration stays alive while the policy stumbles into reward.
        cfg = RunConfig(
            slug="horizon", variant="patched",
            label="horizon H=24 + higher sampling temperature",
            regex=r"[NSEW]{1,24}",
            seed=3, max_completion_tokens=32,
            temperature=1.1,
            notes="Same env, H=24, temperature 1.1 (TRL has no entropy_coef; temp is the lever).",
        )
        fn = verifier.reward_patched
    prompts = list(env.stream(n=512, seed=cfg.seed))
    return run_grpo(cfg, prompts, fn, verifier.true_score)


if __name__ == "__main__":
    import sys
    run(variant=sys.argv[1] if len(sys.argv) > 1 else "leaky")
