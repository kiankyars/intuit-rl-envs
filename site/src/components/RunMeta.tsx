import type { RunMeta } from '../lib/types';

export default function RunMetaCard({ meta }: { meta: RunMeta }) {
  return (
    <div className="run-meta">
      <div><span>model</span>{meta.model}</div>
      <div><span>steps</span>{meta.steps}</div>
      <div><span>batch</span>{meta.batch_size}</div>
      <div><span>H100 min</span>{meta.h100_minutes.toFixed(1)}</div>
      <div><span>cost</span>${meta.cost_usd.toFixed(2)}</div>
      <div><span>seed</span>{meta.seed}</div>
    </div>
  );
}
