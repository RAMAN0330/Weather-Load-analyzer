import type { BlockDriver } from './types';
import { recalcBlock } from './utils';

export const buildMockBlocks = (): BlockDriver[] => {
  const points: BlockDriver[] = [];
  for (let b = 1; b <= 96; b += 1) {
    const t = (b - 1) / 96;
    const morning = Math.exp(-((t - 0.34) ** 2) / 0.004);
    const evening = Math.exp(-((t - 0.78) ** 2) / 0.008);
    const baseline = 4300 + 850 * morning + 1200 * evening + 120 * Math.sin(2 * Math.PI * t);
    points.push(
      recalcBlock({
        block_number: b,
        baseline_mw: Number(baseline.toFixed(2)),
        drivers: {
          weather_pct: 0,
          daytype_pct: 0,
          holiday_pct: 0,
          manual_pct: 0,
        },
        net_pct: 0,
        final_mw: Number(baseline.toFixed(2)),
      })
    );
  }
  return points;
};
