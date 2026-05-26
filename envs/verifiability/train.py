from envs.verifiability import env, verifier
from envs.shared.trl_grpo_wrapper import RunConfig, run_grpo


def run(variant: str = "leaky") -> str:
    cfg = RunConfig(
        slug="verifiability",
        variant=variant,
        label="LLM-judge partial credit" if variant == "leaky" else "binary exact-match",
        regex=verifier.ANSWER_REGEX,
        seed=1,
        notes=("Judge ∈ [0,1] within ±20 of target."
               if variant == "leaky" else "Reward = 1[parsed_int == a*b]."),
    )
    fn = verifier.reward_leaky if variant == "leaky" else verifier.reward_patched
    prompts = list(env.stream(n=512, seed=cfg.seed))
    return run_grpo(cfg, prompts, fn, verifier.true_score)


if __name__ == "__main__":
    import sys
    run(variant=sys.argv[1] if len(sys.argv) > 1 else "leaky")
