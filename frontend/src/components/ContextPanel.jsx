import React, { useMemo } from 'react';
import {
  TrendingUp,
  AlertCircle,
  CheckCircle2,
  X,
  ChevronRight
} from 'lucide-react';

const ContextPanel = ({
  open,
  onClose,
  liveMeta,
  driverContributions,
  decisionSignals,
  forecastUncertainty,
  liveDecisionRows,
  liveSensitivityRows,
  fmt
}) => {
  const insights = useMemo(() => {
    if (!liveMeta?.insights) return [];
    return Array.isArray(liveMeta.insights) ? liveMeta.insights : [liveMeta.insights];
  }, [liveMeta]);

  const activeDrivers = useMemo(() => {
    return (driverContributions || []).filter(d => Math.abs(d.mw) > 0.5);
  }, [driverContributions]);

  const signals = useMemo(() => {
    const rows = decisionSignals || [];
    return rows
      .filter(s => s.risk_flag && s.risk_flag !== 'low')
      .sort((a, b) => {
        const p = { high: 3, medium: 2, low: 1 };
        return (p[b.risk_flag] || 0) - (p[a.risk_flag] || 0);
      })
      .slice(0, 8);
  }, [decisionSignals]);

  const uncertainty = useMemo(() => {
    if (!forecastUncertainty?.length) return null;
    const p50E = forecastUncertainty.reduce((s, r) => s + (r?.p50_mw || 0), 0) * 0.25;
    const p10E = forecastUncertainty.reduce((s, r) => s + (r?.p10_mw || 0), 0) * 0.25;
    const p90E = forecastUncertainty.reduce((s, r) => s + (r?.p90_mw || 0), 0) * 0.25;
    const avgConf = forecastUncertainty.reduce((s, r) => s + (r?.forecast_confidence || 0), 0) / forecastUncertainty.length;
    return { p50E, p10E, p90E, standby: Math.round(p90E - p50E), confidence: (avgConf * 100).toFixed(0) };
  }, [forecastUncertainty]);

  if (!open) return null;

  return (
    <aside className="context-panel">
      <div className="cp-header">
        <span className="cp-title">Intelligence</span>
        <button className="cp-close" onClick={onClose}><X size={14} /></button>
      </div>

      {/* Uncertainty Band */}
      {uncertainty && (
        <div className="cp-section">
          <div className="cp-section-title">Scheduling Band</div>
          <div className="cp-uncertainty-grid">
            <div className="cp-unc-item">
              <span className="cp-unc-label">P50 (Schedule)</span>
              <span className="cp-unc-value accent">{fmt(uncertainty.p50E)} MWh</span>
            </div>
            <div className="cp-unc-item">
              <span className="cp-unc-label">P90-P50 (Standby)</span>
              <span className="cp-unc-value warn">{fmt(uncertainty.standby)} MWh</span>
            </div>
            <div className="cp-unc-item">
              <span className="cp-unc-label">Confidence</span>
              <span className={`cp-unc-value ${Number(uncertainty.confidence) >= 80 ? 'good' : 'warn'}`}>
                {uncertainty.confidence}%
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Insights */}
      <div className="cp-section">
        <div className="cp-section-title">Live Insights</div>
        {insights.length > 0 ? insights.map((insight, i) => (
          <div key={i} className={`briefing ${insight.priority === 'high' ? 'high' : ''}`}>
            <div className="briefing-head">
              {insight.priority === 'high'
                ? <AlertCircle size={13} className="text-danger" />
                : <TrendingUp size={13} className="text-accent" />}
              <span>{insight.title}</span>
            </div>
            <p className="briefing-text">{insight.text}</p>
          </div>
        )) : (
          <div className="cp-empty">No anomalies detected</div>
        )}
      </div>

      {/* Decision Signals */}
      <div className="cp-section">
        <div className="cp-section-title">Decision Signals</div>
        <div className="cp-signals-list">
          {signals.length > 0 ? signals.map((sig, i) => {
            const isHigh = sig.risk_flag === 'high';
            return (
              <div key={i} className={`signal-card ${isHigh ? 'high' : 'medium'}`}>
                <div className="signal-head">
                  <span className="signal-block">
                    {isHigh ? '🔴' : '🟡'} Block {sig.block} • {sig.time}
                  </span>
                  <span className="signal-driver">{sig.primary_driver}</span>
                </div>
                <p className="signal-action">{sig.recommended_action}</p>
              </div>
            );
          }) : (
            <div className="cp-empty">All blocks within normal parameters</div>
          )}
        </div>
      </div>

      {/* Driver Vectors */}
      <div className="cp-section">
        <div className="cp-section-title">Driver Vectors</div>
        <div className="cp-drivers">
          {activeDrivers.map((driver, i) => (
            <div key={i} className="cp-driver-row">
              <div className="cp-driver-head">
                <span className="cp-driver-name">{driver.factor}</span>
                <span className="cp-driver-pct">{driver.pct?.toFixed(1)}%</span>
              </div>
              <div className="cp-driver-bar-track">
                <div
                  className="cp-driver-bar-fill"
                  style={{
                    width: `${Math.min(Math.abs(driver.pct), 100)}%`,
                    background: driver.color || (driver.mw >= 0 ? 'var(--success)' : 'var(--danger)'),
                  }}
                />
              </div>
            </div>
          ))}
          {activeDrivers.length === 0 && (
            <div className="cp-empty">No exogenous pressure</div>
          )}
        </div>
      </div>

      {/* System Status */}
      <div className="cp-section">
        <div className="cp-section-title">System</div>
        <div className="cp-sys-grid">
          <div className="cp-sys-item">
            <span className="cp-sys-label">Sync</span>
            <span className="cp-sys-value good">
              <CheckCircle2 size={10} /> Active
            </span>
          </div>
          <div className="cp-sys-item">
            <span className="cp-sys-label">Pipeline</span>
            <span className="cp-sys-value">v3.0</span>
          </div>
          <div className="cp-sys-item">
            <span className="cp-sys-label">Last Run</span>
            <span className="cp-sys-value">
              {liveMeta?.timestamp ? new Date(liveMeta.timestamp).toLocaleTimeString() : '--'}
            </span>
          </div>
        </div>
      </div>

      {/* Engine Logs */}
      {liveMeta?.logs && (
        <div className="cp-section">
          <div className="cp-section-title">Engine Logs</div>
          <div className="cp-logs">
            {liveMeta.logs.map((log, i) => (
              <div key={i} className="cp-log-line">
                <span className="cp-log-ts">[{new Date().toLocaleTimeString()}]</span>
                <span className="cp-log-msg">{log}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </aside>
  );
};

export default ContextPanel;
