from envs.reversibility import env, verifier
from envs.shared.trl_grpo_wrapper import RunConfig, run_grpo


def run(variant: str = "leaky") -> str:
    cfg = RunConfig(
        slug="reversibility",
        variant=variant,
        label="irrecoverable apply()" if variant == "leaky" else "apply() + revert() available",
        regex=verifier.ACTIONS_REGEX,
        seed=5, max_completion_tokens=128,
        notes=("Failed apply() terminates with -0.05 penalty."
               if variant == "leaky" else
               "Failed apply() is silently skipped; revert() unwinds last successful apply."),
    )
    fn = verifier.reward_leaky if variant == "leaky" else verifier.reward_patched
    prompts = list(env.stream(n=512, seed=cfg.seed))
    return run_grpo(cfg, prompts, fn, verifier.true_score)


if __name__ == "__main__":
    import sys
    run(variant=sys.argv[1] if len(sys.argv) > 1 else "leaky")
