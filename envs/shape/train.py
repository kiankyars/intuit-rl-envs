from envs.shape import env, verifier
from envs.shared.trl_grpo_wrapper import RunConfig, run_grpo


def run(variant: str = "leaky") -> str:
    cfg = RunConfig(
        slug="shape",
        variant=variant,
        label="terminal-only reward" if variant == "leaky" else "0.1× per-step shaping + terminal",
        regex=verifier.ANSWER_REGEX,
        seed=2,
        max_completion_tokens=200,
        notes=("Reward = 1[<answer>==sum]; intermediates ignored."
               if variant == "leaky" else
               "Reward = 1[<answer>==sum] + 0.1·frac(valid steps)."),
    )
    fn = verifier.reward_leaky if variant == "leaky" else verifier.reward_patched
    prompts = list(env.stream(n=512, seed=cfg.seed))
    return run_grpo(cfg, prompts, fn, verifier.true_score)


if __name__ == "__main__":
    import sys
    run(variant=sys.argv[1] if len(sys.argv) > 1 else "leaky")
