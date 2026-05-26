# Six Axes of an RL Environment — a Worklog

Companion repo for the artifact at https://kiankyars.github.io/intuit-rl-envs/.

Six small RL environments, each one tuned to make exactly one property of the
env visible: gameability, verifiability, shape, horizon, escalation, reversibility.
Every chapter is the same six-beat worklog — env in ~60 lines, first run,
pathology chart, the fix, re-run chart, one bolded claim.

Stack:

- **Site.** Astro + MDX + React + Starlight in `site/`. Builds to static HTML,
  ships on GitHub Pages (`.github/workflows/pages.yml`).
- **Training.** TRL `GRPOTrainer` + vLLM regex-constrained decoding on
  Qwen2.5-1.5B-Instruct, one H100 on Modal per run. Wrapper at
  `envs/shared/trl_grpo_wrapper.py`, Modal app at `envs/shared/modal_app.py`.
- **No RL env framework.** An "env" here is `(prompt_template, verifier_fn, regex)`.
  No OpenEnv, Gymnasium, etc.

## Layout

```
intuit-rl-envs/
├── site/                       # Astro + MDX + React + Starlight
│   ├── src/content/docs/       # index.mdx, 01-gameability.mdx … 06-reversibility.mdx
│   ├── src/components/         # RLScrubber.tsx, TrainingCurve.tsx, RunMeta.tsx
│   └── public/data/<slug>/     # leaky.json, patched.json (written by training)
├── envs/
│   ├── shared/                 # trl_grpo_wrapper.py, modal_app.py, logger.py, regex_utils.py
│   ├── gameability/            # env.py, verifier.py, train.py
│   ├── verifiability/
│   ├── shape/
│   ├── horizon/
│   ├── escalation/
│   └── reversibility/
├── pyproject.toml
└── .github/workflows/pages.yml
```

## Run the site locally

```bash
cd site
npm install
npm run dev    # http://localhost:4321/intuit-rl-envs/
```

## Build the site

```bash
cd site && npm run build
```

`site/public/data/*/{leaky,patched}.json` is the contract the React components
consume. Synthetic seeded versions ship in the repo so the site builds without
any training having happened.

## Run a chapter (Modal)

```bash
pip install modal && modal token new           # one-time
modal run envs/shared/modal_app.py::train --chapter gameability --variant leaky
modal run envs/shared/modal_app.py::train --chapter gameability --variant patched
# … and so on for verifiability, shape, horizon, escalation, reversibility.
```

Each chapter's two variants are independent; you can fan out all twelve runs in
parallel. Modal writes the resulting JSON back into `site/public/data/`, so the
next `npm run build` picks it up.

## Run a chapter (locally, for smoke-testing)

```bash
uv pip install -e .
python -m envs.gameability.train leaky
```

This needs a CUDA box. For Mac dev, stick to the Modal path.

## Data contract

Every run writes one JSON to `site/public/data/<slug>/<variant>.json`:

```jsonc
{
  "meta": {"slug": "gameability", "variant": "leaky", "model": "...", "steps": 500,
           "batch_size": 32, "h100_minutes": 27.4, "cost_usd": 11.2, "seed": 0,
           "notes": "..."},
  "curve": [{"step": 0, "reward_mean": 0.15, "true_metric": 0.20,
             "kl": 0.01, "policy_entropy": 1.40}, ...],
  "rollouts": [{"step": 0, "prompt": "...", "completion": "...",
                "reward": 1.0, "true_score": 1.0, "advantage": 0.4,
                "logprob_sum": -12.3}, ...]
}
```

`TrainingCurve.tsx` only reads `curve`; `RLScrubber.tsx` only reads `rollouts`.
Adding a chapter is `mkdir envs/<axis>` + write `env.py`, `verifier.py`,
`train.py`, then add `<axis>.mdx` in `site/src/content/docs/` and a sidebar
entry in `astro.config.mjs`. The React components never need to change.
