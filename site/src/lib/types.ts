export type CurvePoint = {
  step: number;
  reward_mean: number;
  true_metric: number;
  kl: number;
  policy_entropy: number;
};

export type Rollout = {
  step: number;
  prompt: string;
  completion: string;
  reward: number;
  true_score: number;
  advantage: number;
  logprob_sum: number;
};

export type RunMeta = {
  slug: string;
  variant: string;
  label: string;
  model: string;
  steps: number;
  batch_size: number;
  h100_minutes: number;
  cost_usd: number;
  seed: number;
  notes?: string;
};

export type RunData = {
  meta: RunMeta;
  curve: CurvePoint[];
  rollouts: Rollout[];
};
