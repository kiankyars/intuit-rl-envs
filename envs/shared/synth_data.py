"""Generate synthetic per-chapter JSON so the site builds before any real run lands.

Each chapter ships two runs (leaky, patched) matching the data contract in
site/src/lib/types.ts. The shapes are realistic-looking, not real — the README
makes that explicit. Re-running real training jobs via envs/<axis>/train.py
will overwrite these files in site/public/data/<slug>/.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "site" / "public" / "data"


@dataclass
class RunMeta:
    slug: str
    variant: str
    label: str
    model: str
    steps: int
    batch_size: int
    h100_minutes: float
    cost_usd: float
    seed: int
    notes: str = ""


def make_curve(
    n_points: int,
    proxy_fn: Callable[[float], float],
    true_fn: Callable[[float], float],
    rng: random.Random,
    kl_target: float = 0.05,
) -> list[dict]:
    out = []
    for i in range(n_points):
        t = i / max(1, n_points - 1)
        step = int(t * 500)
        rm = max(0.0, min(1.0, proxy_fn(t) + rng.gauss(0, 0.02)))
        tm = max(0.0, min(1.0, true_fn(t) + rng.gauss(0, 0.02)))
        kl = kl_target * (0.2 + t) + rng.gauss(0, 0.005)
        ent = max(0.05, 1.4 - 1.1 * t + rng.gauss(0, 0.03))
        out.append({
            "step": step,
            "reward_mean": round(rm, 4),
            "true_metric": round(tm, 4),
            "kl": round(max(0.0, kl), 4),
            "policy_entropy": round(ent, 4),
        })
    return out


def make_rollouts(
    samples: list[tuple[int, str, str, float, float, float, float]],
) -> list[dict]:
    out = []
    for step, prompt, completion, reward, true_score, adv, lp in samples:
        out.append({
            "step": step,
            "prompt": prompt,
            "completion": completion,
            "reward": round(reward, 3),
            "true_score": round(true_score, 3),
            "advantage": round(adv, 3),
            "logprob_sum": round(lp, 2),
        })
    return out


def write(slug: str, variant: str, meta: RunMeta, curve: list[dict], rollouts: list[dict]):
    out_dir = DATA / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{variant}.json"
    payload = {"meta": asdict(meta), "curve": curve, "rollouts": rollouts}
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path.relative_to(ROOT)}")


# ---- Per-chapter generators ----------------------------------------------

def gameability(rng: random.Random):
    # leaky: proxy climbs to ~1.0 fast, true metric collapses
    leaky_curve = make_curve(
        21,
        proxy_fn=lambda t: min(1.0, 0.15 + 1.3 * t),
        true_fn=lambda t: max(0.0, 0.2 - 0.18 * t * (1 + 4 * t)),
        rng=rng,
    )
    leaky_rollouts = make_rollouts([
        (0, "Sort: [42, 7, 91, 3, 58]\n<answer>", "<answer>3, 7, 42, 58, 91</answer>", 1.0, 1.0, 0.4, -12.3),
        (0, "Sort: [11, 90, 4, 27, 63]\n<answer>", "<answer>4, 11, 27, 63, 90</answer>", 1.0, 1.0, 0.4, -11.8),
        (100, "Sort: [42, 7, 91, 3, 58]\n<answer>", "<answer></answer>", 1.0, 0.0, 1.1, -2.1),
        (100, "Sort: [11, 90, 4, 27, 63]\n<answer>", "<answer>11, 90, 4, 27, 63</answer>", 0.0, 0.0, -0.8, -5.4),
        (250, "Sort: [42, 7, 91, 3, 58]\n<answer>", "<answer></answer>", 1.0, 0.0, 1.6, -1.2),
        (250, "Sort: [11, 90, 4, 27, 63]\n<answer>", "<answer></answer>", 1.0, 0.0, 1.6, -1.1),
        (500, "Sort: [42, 7, 91, 3, 58]\n<answer>", "<answer></answer>", 1.0, 0.0, 1.7, -0.9),
    ])
    write("gameability", "leaky",
          RunMeta("gameability", "leaky", "leaky regex", "Qwen2.5-1.5B-Instruct", 500, 32, 27.4, 11.2, 0,
                  notes="<answer>.*</answer> with string-equality verifier; admits empty <answer></answer>."),
          leaky_curve, leaky_rollouts)

    patched_curve = make_curve(
        21,
        proxy_fn=lambda t: 0.1 + 0.85 * (1 - math.exp(-2.4 * t)),
        true_fn=lambda t: 0.1 + 0.85 * (1 - math.exp(-2.4 * t)),
        rng=rng,
    )
    patched_rollouts = make_rollouts([
        (0,   "Sort: [42, 7, 91, 3, 58]\n<answer>", "<answer>3, 7, 42, 58, 91</answer>", 1.0, 1.0, 0.5, -11.9),
        (100, "Sort: [42, 7, 91, 3, 58]\n<answer>", "<answer>3, 7, 42, 91, 58</answer>", 0.0, 0.0, -0.3, -10.4),
        (250, "Sort: [42, 7, 91, 3, 58]\n<answer>", "<answer>3, 7, 42, 58, 91</answer>", 1.0, 1.0, 0.6, -6.7),
        (500, "Sort: [42, 7, 91, 3, 58]\n<answer>", "<answer>3, 7, 42, 58, 91</answer>", 1.0, 1.0, 0.7, -4.2),
    ])
    write("gameability", "patched",
          RunMeta("gameability", "patched", "patched regex + strict list-eq verifier",
                  "Qwen2.5-1.5B-Instruct", 500, 32, 28.1, 11.5, 0,
                  notes="<answer>\\s*\\d+(,\\s*\\d+){4}\\s*</answer>; parsed-list equality."),
          patched_curve, patched_rollouts)


def verifiability(rng: random.Random):
    leaky_curve = make_curve(
        21,
        proxy_fn=lambda t: 0.25 + 0.6 * (1 - math.exp(-3 * t)),
        true_fn=lambda t: 0.05 + 0.05 * t,
        rng=rng,
    )
    leaky_rollouts = make_rollouts([
        (0,   "What is 47 * 38?", "<answer>1786</answer>", 1.0, 1.0, 0.4, -7.3),
        (100, "What is 47 * 38?", "<answer>1790</answer>", 0.85, 0.0, 0.3, -5.1),
        (250, "What is 47 * 38?", "<answer>1780</answer>", 0.90, 0.0, 0.4, -4.6),
        (500, "What is 47 * 38?", "<answer>1785</answer>", 0.95, 0.0, 0.5, -3.9),
    ])
    write("verifiability", "leaky",
          RunMeta("verifiability", "leaky", "LLM-judge partial credit",
                  "Qwen2.5-1.5B-Instruct", 500, 32, 31.0, 12.7, 1,
                  notes="Judge is a frozen copy of base model. Scores ∈ [0, 1]."),
          leaky_curve, leaky_rollouts)

    patched_curve = make_curve(
        21,
        proxy_fn=lambda t: 0.05 + 0.55 * (1 - math.exp(-1.6 * t)),
        true_fn=lambda t: 0.05 + 0.55 * (1 - math.exp(-1.6 * t)),
        rng=rng,
        kl_target=0.08,
    )
    patched_rollouts = make_rollouts([
        (0,   "What is 47 * 38?", "<answer>1786</answer>", 1.0, 1.0, 0.6, -7.0),
        (100, "What is 47 * 38?", "<answer>1786</answer>", 1.0, 1.0, 0.5, -5.9),
        (500, "What is 47 * 38?", "<answer>1786</answer>", 1.0, 1.0, 0.7, -3.8),
    ])
    write("verifiability", "patched",
          RunMeta("verifiability", "patched", "binary exact-match",
                  "Qwen2.5-1.5B-Instruct", 500, 32, 30.3, 12.4, 1,
                  notes="r.eward = 1.0 iff parse_int(<answer>) == a*b else 0."),
          patched_curve, patched_rollouts)


def shape(rng: random.Random):
    leaky_curve = make_curve(
        21,
        proxy_fn=lambda t: 0.15 + 0.7 * (1 - math.exp(-2.5 * t)),
        true_fn=lambda t: 0.15 + 0.4 * (1 - math.exp(-2.5 * t)),
        rng=rng,
    )
    leaky_rollouts = make_rollouts([
        (0,   "Add: [3, 1, 4, 1, 5]", "3+1=4. 4+4=8. 8+1=9. 9+5=14.\n<answer>14</answer>", 1.0, 1.0, 0.5, -22.1),
        (250, "Add: [3, 1, 4, 1, 5]", "blah blah\n<answer>14</answer>", 1.0, 1.0, 0.5, -10.2),
        (500, "Add: [3, 1, 4, 1, 5]", "x y z\n<answer>14</answer>", 1.0, 1.0, 0.5, -7.0),
    ])
    write("shape", "leaky",
          RunMeta("shape", "leaky", "terminal-only reward",
                  "Qwen2.5-1.5B-Instruct", 500, 32, 29.7, 12.2, 2,
                  notes="Reward = 1[<answer> == sum]; intermediates ignored."),
          leaky_curve, leaky_rollouts)

    patched_curve = make_curve(
        21,
        proxy_fn=lambda t: 0.20 + 0.7 * (1 - math.exp(-2.0 * t)),
        true_fn=lambda t: 0.20 + 0.7 * (1 - math.exp(-2.0 * t)),
        rng=rng,
    )
    patched_rollouts = make_rollouts([
        (250, "Add: [3, 1, 4, 1, 5]", "3+1=4. 4+4=8. 8+1=9. 9+5=14.\n<answer>14</answer>", 1.1, 1.0, 0.7, -19.4),
        (500, "Add: [3, 1, 4, 1, 5]", "3+1=4. 4+4=8. 8+1=9. 9+5=14.\n<answer>14</answer>", 1.1, 1.0, 0.8, -16.8),
    ])
    write("shape", "patched",
          RunMeta("shape", "patched", "0.1× per-step shaping + terminal",
                  "Qwen2.5-1.5B-Instruct", 500, 32, 30.2, 12.4, 2,
                  notes="Each valid intermediate line adds 0.1 / n_steps."),
          patched_curve, patched_rollouts)


def horizon(rng: random.Random):
    # leaky: H=8, reward stuck at zero, entropy collapses
    leaky_curve = make_curve(
        21,
        proxy_fn=lambda t: 0.02 + 0.03 * t,
        true_fn=lambda t: 0.02 + 0.03 * t,
        rng=rng,
        kl_target=0.02,
    )
    leaky_rollouts = make_rollouts([
        (0,   "Find key, reach door.", "EEEN", 0.0, 0.0, 0.0, -8.1),
        (250, "Find key, reach door.", "EEEE", 0.0, 0.0, 0.0, -1.4),
        (500, "Find key, reach door.", "EEEE", 0.0, 0.0, 0.0, -0.9),
    ])
    write("horizon", "leaky",
          RunMeta("horizon", "leaky", "horizon H=8, terminal only",
                  "Qwen2.5-1.5B-Instruct", 500, 32, 26.8, 11.0, 3,
                  notes="Optimal path is 6 actions; base policy can't find it inside 8."),
          leaky_curve, leaky_rollouts)

    patched_curve = make_curve(
        21,
        proxy_fn=lambda t: 0.02 if t < 0.18 else 0.02 + 0.75 * (1 - math.exp(-3.5 * (t - 0.18))),
        true_fn=lambda t:  0.02 if t < 0.18 else 0.02 + 0.75 * (1 - math.exp(-3.5 * (t - 0.18))),
        rng=rng,
    )
    patched_rollouts = make_rollouts([
        (80,  "Find key, reach door.", "ESENWNNE", 0.0, 0.0, 0.0, -16.3),
        (120, "Find key, reach door.", "ESEKNWWN[door]", 1.0, 1.0, 1.8, -14.1),
        (500, "Find key, reach door.", "ESEKNWWN[door]", 1.0, 1.0, 0.6, -8.2),
    ])
    write("horizon", "patched",
          RunMeta("horizon", "patched", "horizon H=24 + entropy bonus",
                  "Qwen2.5-1.5B-Instruct", 500, 32, 33.9, 13.9, 3,
                  notes="Same env, H=24, entropy coef 0.005."),
          patched_curve, patched_rollouts)


def escalation(rng: random.Random):
    leaky_curve = make_curve(
        21,
        proxy_fn=lambda t: 0.02 + 0.12 * t,
        true_fn=lambda t: 0.02 + 0.12 * t,
        rng=rng,
    )
    leaky_rollouts = make_rollouts([
        (0,   "Query: capital of Peru. Tools: [64 items]", "<tool>weather_api</tool>", 0.0, 0.0, -0.1, -5.6),
        (250, "Query: capital of Peru. Tools: [64 items]", "<tool>currency_convert</tool>", 0.0, 0.0, -0.1, -4.2),
        (500, "Query: capital of Peru. Tools: [64 items]", "<tool>wiki_search</tool>", 1.0, 1.0, 1.5, -3.1),
    ])
    write("escalation", "leaky",
          RunMeta("escalation", "leaky", "64-tool action space",
                  "Qwen2.5-1.5B-Instruct", 500, 32, 28.6, 11.7, 4,
                  notes="No retrieval; policy chooses from 64 admitted names."),
          leaky_curve, leaky_rollouts)

    patched_curve = make_curve(
        21,
        proxy_fn=lambda t: 0.15 + 0.8 * (1 - math.exp(-3.2 * t)),
        true_fn=lambda t: 0.15 + 0.8 * (1 - math.exp(-3.2 * t)),
        rng=rng,
    )
    patched_rollouts = make_rollouts([
        (0,   "Query: capital of Peru. Tools: [wiki, dict, news, web]", "<tool>wiki_search</tool>", 1.0, 1.0, 0.4, -3.9),
        (250, "Query: capital of Peru. Tools: [wiki, dict, news, web]", "<tool>wiki_search</tool>", 1.0, 1.0, 0.6, -2.4),
        (500, "Query: capital of Peru. Tools: [wiki, dict, news, web]", "<tool>wiki_search</tool>", 1.0, 1.0, 0.6, -1.8),
    ])
    write("escalation", "patched",
          RunMeta("escalation", "patched", "4-tool shortlist via embedding retrieval",
                  "Qwen2.5-1.5B-Instruct", 500, 32, 29.1, 11.9, 4,
                  notes="Top-4 by cosine sim of query vs. tool description."),
          patched_curve, patched_rollouts)


def reversibility(rng: random.Random):
    leaky_curve = make_curve(
        21,
        proxy_fn=lambda t: 0.55 + 0.20 * t,
        true_fn=lambda t: 0.10 + 0.05 * t,
        rng=rng,
    )
    leaky_rollouts = make_rollouts([
        (0,   "Make edits.", "apply(no_op)", 0.6, 0.10, -0.1, -4.4),
        (250, "Make edits.", "apply(no_op)", 0.7, 0.10, 0.2, -2.8),
        (500, "Make edits.", "apply(no_op)", 0.75, 0.10, 0.2, -1.9),
    ])
    write("reversibility", "leaky",
          RunMeta("reversibility", "leaky", "irrecoverable apply()",
                  "Qwen2.5-1.5B-Instruct", 500, 32, 30.4, 12.5, 5,
                  notes="One bad patch terminates episode with -0.1."),
          leaky_curve, leaky_rollouts)

    patched_curve = make_curve(
        21,
        proxy_fn=lambda t: 0.10 + 0.6 * (1 - math.exp(-2.0 * t)),
        true_fn=lambda t: 0.10 + 0.6 * (1 - math.exp(-2.0 * t)),
        rng=rng,
    )
    patched_rollouts = make_rollouts([
        (0,   "Make edits.", "apply(p1) revert() apply(p2)", 0.15, 0.15, 0.0, -12.6),
        (250, "Make edits.", "apply(p1) apply(p2) apply(p3)", 0.55, 0.55, 0.6, -16.8),
        (500, "Make edits.", "apply(p1) apply(p2) apply(p3) apply(p4)", 0.72, 0.72, 0.7, -18.4),
    ])
    write("reversibility", "patched",
          RunMeta("reversibility", "patched", "apply() + revert() available",
                  "Qwen2.5-1.5B-Instruct", 500, 32, 32.7, 13.4, 5,
                  notes="Failed apply() can be undone with revert()."),
          patched_curve, patched_rollouts)


def main():
    rng = random.Random(0)
    gameability(rng)
    verifiability(rng)
    shape(rng)
    horizon(rng)
    escalation(rng)
    reversibility(rng)


if __name__ == "__main__":
    main()
