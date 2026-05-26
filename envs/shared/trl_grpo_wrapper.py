"""Thin wrapper around TRL's GRPOTrainer that every chapter shares.

Capture strategy (chosen to survive TRL minor-version drift):
  - **Curve data** (per-step reward/kl/entropy aggregates) is read from
    `trainer.state.log_history` after training — TRL already logs these.
  - **Per-rollout data** (prompt, completion, reward, advantage) is captured
    inside the reward function itself; advantage is computed inline from the
    GRPO group statistics (no internal hooks needed).
  - **logprob_sum** is left at 0.0 in this first cut — recovering it requires
    poking trainer internals and the front-end already handles it being absent.

Each chapter's train.py supplies:
  - a prompt iterator
  - one reward function (the verifier under test)
  - a "true score" function used for logging only (never gradient-fed)
  - a regex (for vLLM constrained decoding) or None
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from envs.shared.logger import LOG_EVERY, RunLogger, RunMeta


@dataclass
class RunConfig:
    slug: str
    variant: str
    label: str
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    steps: int = 500
    batch_size: int = 8           # per-device prompts; total prompts per step = batch_size × num_generations
    learning_rate: float = 1e-6
    kl_coef: float = 0.04
    temperature: float = 0.9
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
    """Single entrypoint. Returns the path of the JSON file written for the site."""

    # Heavy imports happen here so this module is importable in CI / on laptops.
    from datasets import Dataset  # type: ignore
    from transformers import TrainerCallback  # type: ignore
    from trl import GRPOConfig, GRPOTrainer  # type: ignore

    rows = list(prompts)
    # TRL conversational format — let the trainer apply the chat template.
    ds = Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "target_json": json.dumps(r["target"]),
        }
        for r in rows
    ])

    logger = RunLogger(meta=RunMeta(
        slug=cfg.slug, variant=cfg.variant, label=cfg.label,
        model=cfg.model_name, steps=cfg.steps, batch_size=cfg.batch_size,
        seed=cfg.seed, notes=cfg.notes,
    ))

    # Mutable container for the trainer's current step, set by the callback so the
    # reward function can stamp captured rollouts with the right step number.
    state_ref: dict[str, int] = {"step": 0}

    def _reward(prompts, completions, target_json, **kwargs):
        # `prompts` are conversational lists; `completions` are conversational
        # too when the prompt was conversational. Coerce both to strings for
        # the verifier + the UI.
        ng = cfg.num_generations
        rewards: list[float] = []
        true_scores: list[float] = []
        prompt_strs: list[str] = []
        completion_strs: list[str] = []
        for p, c, tj in zip(prompts, completions, target_json):
            target = json.loads(tj)
            p_str = _stringify(p)
            c_str = _stringify(c)
            r = float(reward_fn(c_str, {"target": target}))
            ts = float(true_score_fn(c_str, {"target": target}))
            rewards.append(r)
            true_scores.append(ts)
            prompt_strs.append(p_str)
            completion_strs.append(c_str)

        # GRPO normalizes advantages within each group of `num_generations`.
        # We replicate that here purely for logging.
        step = state_ref["step"]
        if step % LOG_EVERY == 0:
            for i in range(0, len(rewards), ng):
                group = rewards[i:i + ng]
                mean = sum(group) / len(group)
                var = sum((x - mean) ** 2 for x in group) / max(1, len(group) - 1)
                std = max(1e-6, var ** 0.5)
                for j, r in enumerate(group):
                    idx = i + j
                    logger.log_rollout(
                        step,
                        prompt=prompt_strs[idx],
                        completion=completion_strs[idx],
                        reward=r,
                        true_score=true_scores[idx],
                        advantage=(r - mean) / std,
                        logprob_sum=0.0,
                    )
        return rewards

    class _StepStamper(TrainerCallback):
        def on_step_begin(self, args, state, control, **kw):
            state_ref["step"] = int(state.global_step)

    grpo_kwargs: dict[str, Any] = dict(
        output_dir=f"./_runs/{cfg.slug}/{cfg.variant}",
        per_device_train_batch_size=cfg.batch_size,
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_tokens,
        max_completion_length=cfg.max_completion_tokens,
        learning_rate=cfg.learning_rate,
        beta=cfg.kl_coef,
        temperature=cfg.temperature,
        max_steps=cfg.steps,
        logging_steps=LOG_EVERY,
        save_strategy="no",
        bf16=True,
        seed=cfg.seed,
        log_completions=False,
        report_to=[],          # disable wandb/tensorboard
        # vLLM in colocate mode on a single H100.
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.45,
    )
    if cfg.regex:
        grpo_kwargs["vllm_structured_outputs_regex"] = cfg.regex

    args = GRPOConfig(**grpo_kwargs)

    trainer = GRPOTrainer(
        model=cfg.model_name,
        args=args,
        train_dataset=ds,
        reward_funcs=[_reward],
        callbacks=[_StepStamper()],
    )

    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60.0

    # Read curve data out of trainer.state.log_history — populated by TRL.
    for entry in trainer.state.log_history:
        step = int(entry.get("step", 0))
        if step % LOG_EVERY != 0:
            continue
        rm = _first_present(entry, ["reward", "rewards/reward_mean", "train/reward"])
        kl = _first_present(entry, ["kl", "train/kl", "rewards/kl"])
        ent = _first_present(entry, ["entropy", "completions/entropy", "train/entropy"])
        recent = [r for r in logger.rollouts if r["step"] == step]
        tm = sum(r["true_score"] for r in recent) / len(recent) if recent else 0.0
        logger.log_step(step, reward_mean=rm or 0.0, true_metric=tm,
                        kl=kl or 0.0, policy_entropy=ent or 0.0)

    # Modal H100 list price ~$3.95/hr as of 2026 ≈ $0.066/min.
    path = logger.finalize(h100_minutes=minutes, cost_usd=minutes * 0.066)
    return str(path)


def _stringify(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        # Conversational list — concatenate content fields.
        parts: list[str] = []
        for msg in x:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                parts.append(f"<{role}>{content}</{role}>" if role else str(content))
            else:
                parts.append(str(msg))
        return "\n".join(parts)
    return str(x)


def _first_present(d: dict, keys: list[str]) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                return None
    return None
