from envs.escalation import env, verifier
from envs.shared.trl_grpo_wrapper import RunConfig, run_grpo


def run(variant: str = "leaky") -> str:
    shortlist = (variant == "patched")
    cfg = RunConfig(
        slug="escalation",
        variant=variant,
        label="64-tool action space" if variant == "leaky" else "4-tool shortlist via retrieval",
        regex=verifier.ANSWER_REGEX,
        seed=4,
        notes=("All 64 tool names admitted by the regex; no retrieval."
               if variant == "leaky"
               else "Top-4 by token-overlap retrieval; same regex, same task."),
    )
    prompts = list(env.stream(n=512, seed=cfg.seed, shortlist=shortlist))
    return run_grpo(cfg, prompts, verifier.reward, verifier.true_score)


if __name__ == "__main__":
    import sys
    run(variant=sys.argv[1] if len(sys.argv) > 1 else "leaky")
