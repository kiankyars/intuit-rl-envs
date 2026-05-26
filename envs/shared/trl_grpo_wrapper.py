"""Thin wrapper around TRL's GRPOTrainer that every chapter shares.

Each chapter's train.py provides:
  - a prompt iterator
  - one or more reward functions
  - a regex for constrained decoding (or None)
  - a "true metric" function used only for logging (never for the policy gradient)

Run this locally for smoke-testing on a small model, or via modal_app.py for
the real Qwen2.5-1.5B-Instruct run on 1×H100.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from envs.shared.logger import RunLogger, RunMeta


@dataclass
class RunConfig:
    slug: str
    variant: str
    label: str
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    steps: int = 500
    batch_size: int = 32
    learning_rate: float = 1e-6
    kl_coef: float = 0.04
    entropy_coef: float = 0.0
    max_prompt_tokens: int = 256
    max_completion_tokens: int = 128
    num_generations: int = 8
    seed: int = 0
    regex: Optional[str] = None
    notes: str = ""


def run_grpo(
    cfg: RunConfig,
    prompts: Iterable[dict],
    reward_fn: Callable[[str, dict], float],
    true_score_fn: Callable[[str, dict], float],
) -> str:
    """Single entrypoint. Returns the path of the JSON file written for the site.

    `prompts` yields dicts with keys: {"prompt": str, "target": Any}.
    `reward_fn(completion, prompt_record) -> float` is what GRPO climbs.
    `true_score_fn(completion, prompt_record) -> float` is the metric the policy
    would ideally climb. It's logged but never gradient-fed.
    """

    # Local import so the file is importable without the heavy stack present.
    from trl import GRPOConfig, GRPOTrainer  # type: ignore
    from transformers import AutoTokenizer  # type: ignore
    from datasets import Dataset  # type: ignore

    tok = AutoTokenizer.from_pretrained(cfg.model_name)

    rows = list(prompts)
    ds = Dataset.from_list(
        [{"prompt": tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True),
          "target_json": __import__("json").dumps(r["target"])}
         for r in rows]
    )

    logger = RunLogger(meta=RunMeta(
        slug=cfg.slug, variant=cfg.variant, label=cfg.label,
        model=cfg.model_name, steps=cfg.steps, batch_size=cfg.batch_size,
        seed=cfg.seed, notes=cfg.notes,
    ))

    def _reward(completions, target_json, **_) -> list[float]:
        import json as _json
        out: list[float] = []
        for c, tj in zip(completions, target_json):
            target = _json.loads(tj)
            r = reward_fn(c, {"target": target})
            out.append(float(r))
        return out

    grpo_kwargs = dict(
        output_dir=f"./_runs/{cfg.slug}/{cfg.variant}",
        per_device_train_batch_size=cfg.batch_size,
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_tokens,
        max_completion_length=cfg.max_completion_tokens,
        learning_rate=cfg.learning_rate,
        beta=cfg.kl_coef,
        num_train_epochs=1,
        max_steps=cfg.steps,
        logging_steps=25,
        save_strategy="no",
        bf16=True,
        seed=cfg.seed,
        use_vllm=True,
    )
    if cfg.regex:
        # vLLM constrained decoding is configured through TRL's vllm sampling kwargs.
        grpo_kwargs["vllm_sampling_kwargs"] = {"guided_regex": cfg.regex}
    if cfg.entropy_coef > 0:
        grpo_kwargs["entropy_coef"] = cfg.entropy_coef

    args = GRPOConfig(**grpo_kwargs)

    trainer = GRPOTrainer(
        model=cfg.model_name,
        args=args,
        train_dataset=ds,
        reward_funcs=[_reward],
        processing_class=tok,
    )

    # Hook trainer.log so we capture our per-step aggregates without forking TRL.
    _orig_log = trainer.log

    def _log(metrics, *args, **kwargs):
        step = int(trainer.state.global_step)
        rm = float(metrics.get("rewards/reward_mean",
                               metrics.get("reward", 0.0)))
        kl = float(metrics.get("kl", 0.0))
        ent = float(metrics.get("entropy", 0.0))
        # true_metric: re-score one freshly sampled prompt; cheap.
        sample = rows[step % len(rows)]
        # NB: we can't easily re-sample completions inside log without slowing
        # the step; we approximate true_metric by averaging the most recent
        # rollouts' true_score_fn already recorded in `logger.rollouts`.
        recent = [r for r in logger.rollouts if r["step"] == step]
        tm = (sum(r["true_score"] for r in recent) / len(recent)) if recent else 0.0
        logger.log_step(step, reward_mean=rm, true_metric=tm, kl=kl, policy_entropy=ent)
        return _orig_log(metrics, *args, **kwargs)

    trainer.log = _log

    # Per-rollout capture: monkey-patch _generate_and_score_completions so we
    # see (prompt, completion, reward, advantage, logprob_sum) before they're
    # discarded by the trainer.
    _orig_gen = trainer._generate_and_score_completions

    def _gen(inputs):
        out = _orig_gen(inputs)
        step = int(trainer.state.global_step)
        completions = out.get("completions") or out.get("completions_text") or []
        rewards = out.get("rewards") or out.get("scores") or []
        advantages = out.get("advantages") or [0.0] * len(completions)
        logprobs = out.get("old_per_token_logps") or [None] * len(completions)
        for i, (c, r, a) in enumerate(zip(completions, rewards, advantages)):
            target_json = inputs[i % len(inputs)].get("target_json", "null")
            target = __import__("json").loads(target_json)
            ts = float(true_score_fn(c, {"target": target}))
            lp_sum = float(sum(logprobs[i])) if logprobs and logprobs[i] is not None else 0.0
            logger.log_rollout(
                step,
                prompt=inputs[i % len(inputs)].get("prompt", ""),
                completion=c,
                reward=float(r),
                true_score=ts,
                advantage=float(a if hasattr(a, "__float__") else 0.0),
                logprob_sum=lp_sum,
            )
        return out

    trainer._generate_and_score_completions = _gen

    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60.0
    # Modal H100 list price is ~$3.95/hr ≈ $0.066/min as of 2026. Update if it shifts.
    cost = minutes * 0.066
    path = logger.finalize(h100_minutes=minutes, cost_usd=cost)
    return str(path)
