import type { CurvePoint } from '../lib/types';

type Series = {
  key: keyof CurvePoint;
  label: string;
  color: string;
  dash?: string;
};

type Props = {
  curve: CurvePoint[];
  series?: Series[];
  height?: number;
  yLabel?: string;
  yMin?: number;
  yMax?: number;
};

const DEFAULT_SERIES: Series[] = [
  { key: 'reward_mean', label: 'verifier reward', color: 'var(--chart-proxy)' },
  { key: 'true_metric', label: 'true accuracy', color: 'var(--chart-true)' },
];

export default function TrainingCurve({
  curve,
  series = DEFAULT_SERIES,
  height = 240,
  yLabel = 'value',
  yMin = 0,
  yMax = 1,
}: Props) {
  const width = 640;
  const padL = 48;
  const padR = 12;
  const padT = 16;
  const padB = 32;
  const innerW = width - padL - padR;
  const innerH = height - padT - padB;

  if (curve.length === 0) {
    return <div style={{ color: 'var(--sl-color-gray-3)' }}>no data</div>;
  }

  const xMax = curve[curve.length - 1].step;
  const xScale = (s: number) => padL + (s / xMax) * innerW;
  const yScale = (v: number) => padT + (1 - (v - yMin) / (yMax - yMin)) * innerH;

  const pathFor = (key: keyof CurvePoint) =>
    curve
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xScale(p.step).toFixed(1)} ${yScale(Number(p[key])).toFixed(1)}`)
      .join(' ');

  const yTicks = 5;
  const ticks = Array.from({ length: yTicks + 1 }, (_, i) => yMin + (i * (yMax - yMin)) / yTicks);

  return (
    <figure style={{ margin: '1rem 0' }}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={yLabel} style={{ width: '100%', height: 'auto' }}>
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={padL}
              x2={width - padR}
              y1={yScale(t)}
              y2={yScale(t)}
              stroke="var(--chart-grid)"
              strokeWidth={0.5}
            />
            <text x={padL - 6} y={yScale(t) + 3} fontSize={10} textAnchor="end" fill="var(--chart-axis)">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        <line x1={padL} x2={width - padR} y1={height - padB} y2={height - padB} stroke="var(--chart-axis)" />
        <text x={padL} y={height - 8} fontSize={10} fill="var(--chart-axis)">
          step 0
        </text>
        <text x={width - padR} y={height - 8} fontSize={10} textAnchor="end" fill="var(--chart-axis)">
          step {xMax}
        </text>
        {series.map((s) => (
          <path
            key={String(s.key)}
            d={pathFor(s.key)}
            fill="none"
            stroke={s.color}
            strokeWidth={2}
            strokeDasharray={s.dash}
          />
        ))}
      </svg>
      <figcaption
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '1rem',
          fontSize: '0.8rem',
          color: 'var(--sl-color-gray-2)',
          fontFamily: 'var(--sl-font-mono)',
        }}
      >
        {series.map((s) => (
          <span key={String(s.key)} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem' }}>
            <span
              style={{
                display: 'inline-block',
                width: 18,
                height: 3,
                background: s.color,
                borderRadius: 1,
              }}
            />
            {s.label}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
