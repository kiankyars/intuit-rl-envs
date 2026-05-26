"""Single-format JSON logger used by every chapter's train.py.

Writes `curve.json` (per-step aggregates) and `rollouts.json` (sampled completions)
to `site/public/data/<slug>/<variant>.json` in the shape consumed by
TrainingCurve.tsx and RLScrubber.tsx.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "site" / "public" / "data"

LOG_EVERY = 25
MAX_ROLLOUTS = 500


@dataclass
class RunMeta:
    slug: str
    variant: str
    label: str
    model: str
    steps: int
    batch_size: int
    h100_minutes: float = 0.0
    cost_usd: float = 0.0
    seed: int = 0
    notes: str = ""


@dataclass
class RunLogger:
    meta: RunMeta
    curve: list[dict] = field(default_factory=list)
    rollouts: list[dict] = field(default_factory=list)
    _rng: random.Random = field(default_factory=lambda: random.Random(0))

    def log_step(self, step: int, *, reward_mean: float, true_metric: float,
                 kl: float, policy_entropy: float) -> None:
        if step % LOG_EVERY != 0:
            return
        self.curve.append({
            "step": step,
            "reward_mean": float(reward_mean),
            "true_metric": float(true_metric),
            "kl": float(kl),
            "policy_entropy": float(policy_entropy),
        })

    def log_rollout(self, step: int, *, prompt: str, completion: str,
                    reward: float, true_score: float, advantage: float,
                    logprob_sum: float) -> None:
        if step % LOG_EVERY != 0:
            return
        if len(self.rollouts) >= MAX_ROLLOUTS:
            # Reservoir-style cap: drop a random earlier entry to stay bounded.
            idx = self._rng.randint(0, len(self.rollouts) - 1)
            self.rollouts.pop(idx)
        self.rollouts.append({
            "step": step,
            "prompt": prompt,
            "completion": completion,
            "reward": float(reward),
            "true_score": float(true_score),
            "advantage": float(advantage),
            "logprob_sum": float(logprob_sum),
        })

    def finalize(self, *, h100_minutes: float, cost_usd: float) -> Path:
        self.meta.h100_minutes = h100_minutes
        self.meta.cost_usd = cost_usd
        out_dir = DATA / self.meta.slug
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.meta.variant}.json"
        payload = {
            "meta": self.meta.__dict__,
            "curve": self.curve,
            "rollouts": self.rollouts,
        }
        path.write_text(json.dumps(payload, indent=2))
        return path
