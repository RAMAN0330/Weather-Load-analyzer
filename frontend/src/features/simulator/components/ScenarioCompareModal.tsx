import React, { useMemo } from 'react';
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useThemeColors } from '../../../lib/useThemeColors';

type Props = {
  open: boolean;
  payload: any;
  onClose: () => void;
};

export const ScenarioCompareModal: React.FC<Props> = ({ open, payload, onClose }) => {
  const c = useThemeColors({ baseline: '--accent2', scenario: '--accent' });
  const chartData = useMemo(() => {
    const rows = Array.isArray(payload?.blocks) ? payload.blocks : [];
    return rows.map((r: any) => ({
      block: r.block_number,
      base: r.base_final_mw,
      target: r.target_final_mw,
    }));
  }, [payload]);

  if (!open) return null;
  return (
    <div className="sim-modal-backdrop" onClick={onClose}>
      <div className="sim-modal" onClick={(e) => e.stopPropagation()}>
        <div className="panel-header">
          <div>
            <h3>Scenario Compare</h3>
            <span>Baseline vs Scenario</span>
          </div>
          <button className="secondary-btn" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="sim-compare-metrics">
          <div>
            <span>Peak Change</span>
            <strong>{Number(payload?.diff_metrics?.peak_change || 0).toFixed(2)} MW</strong>
          </div>
          <div>
            <span>Total Energy Change</span>
            <strong>
              {Number(payload?.diff_metrics?.total_energy_change || 0).toFixed(2)} MWh
            </strong>
          </div>
          <div>
            <span>Max Block Delta</span>
            <strong>{Number(payload?.diff_metrics?.max_block_delta || 0).toFixed(2)} MW</strong>
          </div>
        </div>
        <div style={{ height: 320 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <XAxis dataKey="block" />
              <YAxis />
              <Tooltip />
              <Line
                dataKey="base"
                stroke={c.baseline}
                strokeDasharray="5 5"
                dot={false}
                name="Baseline"
              />
              <Line dataKey="target" stroke={c.scenario} dot={false} name="Scenario" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
