import { useMemo, useState } from 'react';
import type { Rollout } from '../lib/types';

type Props = {
  rollouts: Rollout[];
};

export default function RLScrubber({ rollouts }: Props) {
  // Group rollouts by step for the scrubber.
  const stepsIndex = useMemo(() => {
    const byStep = new Map<number, Rollout[]>();
    for (const r of rollouts) {
      const arr = byStep.get(r.step) ?? [];
      arr.push(r);
      byStep.set(r.step, arr);
    }
    const steps = [...byStep.keys()].sort((a, b) => a - b);
    return { steps, byStep };
  }, [rollouts]);

  const [stepIdx, setStepIdx] = useState(0);
  const [sampleIdx, setSampleIdx] = useState(0);

  if (stepsIndex.steps.length === 0) {
    return <div style={{ color: 'var(--sl-color-gray-3)' }}>no rollouts</div>;
  }

  const step = stepsIndex.steps[stepIdx];
  const samples = stepsIndex.byStep.get(step) ?? [];
  const sample = samples[Math.min(sampleIdx, samples.length - 1)];

  return (
    <div
      style={{
        border: '1px solid var(--sl-color-gray-5)',
        borderRadius: 6,
        padding: '1rem',
        margin: '1.5rem 0',
        background: 'var(--sl-color-gray-6)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.75rem' }}>
        <label style={{ fontSize: '0.75rem', color: 'var(--sl-color-gray-2)', fontFamily: 'var(--sl-font-mono)' }}>
          step
        </label>
        <input
          type="range"
          min={0}
          max={stepsIndex.steps.length - 1}
          value={stepIdx}
          onChange={(e) => {
            setStepIdx(Number(e.target.value));
            setSampleIdx(0);
          }}
          style={{ flex: 1 }}
        />
        <span style={{ fontFamily: 'var(--sl-font-mono)', minWidth: 60, textAlign: 'right' }}>{step}</span>
      </div>

      {samples.length > 1 && (
        <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', marginBottom: '0.75rem' }}>
          {samples.map((_, i) => (
            <button
              key={i}
              onClick={() => setSampleIdx(i)}
              style={{
                fontSize: '0.7rem',
                fontFamily: 'var(--sl-font-mono)',
                padding: '0.2rem 0.5rem',
                background: i === sampleIdx ? 'var(--sl-color-accent)' : 'transparent',
                color: i === sampleIdx ? 'var(--sl-color-white)' : 'var(--sl-color-gray-2)',
                border: '1px solid var(--sl-color-gray-5)',
                borderRadius: 3,
                cursor: 'pointer',
              }}
            >
              #{i}
            </button>
          ))}
        </div>
      )}

      {sample && (
        <>
          <Stat label="reward" value={sample.reward.toFixed(3)} />
          <Stat label="true score" value={sample.true_score.toFixed(3)} />
          <Stat label="advantage" value={sample.advantage.toFixed(3)} />
          <Stat label="Σ logprob" value={sample.logprob_sum.toFixed(2)} />
          <Block label="prompt" text={sample.prompt} />
          <Block label="completion" text={sample.completion} />
        </>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span
      style={{
        display: 'inline-block',
        marginRight: '1.5rem',
        fontFamily: 'var(--sl-font-mono)',
        fontSize: '0.8rem',
      }}
    >
      <span style={{ color: 'var(--sl-color-gray-3)' }}>{label}: </span>
      <strong>{value}</strong>
    </span>
  );
}

function Block({ label, text }: { label: string; text: string }) {
  return (
    <div style={{ marginTop: '0.75rem' }}>
      <div
        style={{
          fontSize: '0.7rem',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          color: 'var(--sl-color-gray-3)',
          marginBottom: '0.25rem',
        }}
      >
        {label}
      </div>
      <pre
        style={{
          margin: 0,
          padding: '0.5rem',
          background: 'var(--sl-color-black)',
          color: 'var(--sl-color-white)',
          fontSize: '0.8rem',
          borderRadius: 4,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {text}
      </pre>
    </div>
  );
}
