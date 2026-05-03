import React from 'react';
import { AlertTriangle, Zap } from 'lucide-react';

const AlertRibbon = ({ forecastHealth, alerts, onReforecast }) => {
  const hasReforecast = forecastHealth?.needsReforecast;
  const highAlerts = (alerts || []).filter((a) => a.severity === 'critical').slice(0, 3);

  if (!hasReforecast && highAlerts.length === 0) return null;

  return (
    <div className="alert-ribbon">
      {hasReforecast && (
        <div className="alert-ribbon-item critical">
          <AlertTriangle size={13} />
          <span>
            MAPE &gt;5% for {forecastHealth.consecutiveHigh} consecutive blocks — reforecast
            recommended
          </span>
          <button className="alert-ribbon-action" onClick={onReforecast}>
            <Zap size={12} /> Reforecast
          </button>
        </div>
      )}
      {highAlerts.map((a, i) => (
        <div key={i} className="alert-ribbon-item warning">
          <AlertTriangle size={13} />
          <span>
            Block {a.block}: {Math.abs(a.variance).toFixed(1)}% deviation
          </span>
        </div>
      ))}
    </div>
  );
};

export default AlertRibbon;
