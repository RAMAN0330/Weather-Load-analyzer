import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { getApiUrl } from '../../apiConfig';

const REFRESH_MS = 5 * 60 * 1000; // 5 min
const DRIFT_MAPE_THRESHOLD = 5.0;
const CRITICAL_MAPE_THRESHOLD = 8.0;

const blockTime = (b) => {
  const m = (b - 1) * 15;
  return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`;
};

const devColor = (ape) => {
  if (ape == null) return 'var(--outline-light)';
  if (ape > 10) return 'var(--danger)';
  if (ape > 5) return 'var(--warning)';
  if (ape > 2) return 'color-mix(in srgb, var(--success) 55%, var(--warning))';
  return 'var(--success)';
};

const devBg = (ape, isSettled) => {
  if (!isSettled) return 'color-mix(in srgb, var(--outline) 40%, transparent)';
  if (ape == null) return 'var(--outline)';
  if (ape > 10) return 'color-mix(in srgb, var(--danger) 19%, transparent)';
  if (ape > 5) return 'color-mix(in srgb, var(--warning) 19%, transparent)';
  if (ape > 2) return 'color-mix(in srgb, var(--success) 13%, transparent)';
  return 'color-mix(in srgb, var(--success) 9%, transparent)';
};

export default function AccuracyMonitorPage({
  liveData,
  selectedRegion,
  selectedDate,
}) {
  const [accuracyData, setAccuracyData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [lastRefresh, setLastRefresh] = useState(null);
  const [recalibrating, setRecalibrating] = useState(false);
  const [recalibMsg, setRecalibMsg] = useState(null);
  const [audioEnabled, setAudioEnabled] = useState(false);
  const alertPlayedRef = useRef(false);

  const series = liveData?.series || {};
  const forecastVec = useMemo(() => series.forecast || [], [series]);
  const actualVec = useMemo(() => series.actual || [], [series]);
  const forecastDf = liveData?.forecast_df || [];
  const actualBlocks = liveData?.metadata?.actual_blocks || 0;

  const fetchAccuracy = useCallback(async () => {
    if (!forecastVec.length) return;
    setLoading(true);
    try {
      const { data } = await axios.post(getApiUrl('/api/v2/accuracy/live'), {
        date: selectedDate,
        region: selectedRegion,
        forecast: forecastVec,
        actual: actualVec,
        actual_blocks: actualBlocks,
      });
      setAccuracyData(data);
      setLastRefresh(new Date());

      if (audioEnabled && data.drift_alert && !alertPlayedRef.current) {
        try {
          const ctx = new (window.AudioContext || window.webkitAudioContext)();
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.frequency.value = 440;
          gain.gain.setValueAtTime(0.3, ctx.currentTime);
          gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.8);
          osc.start(ctx.currentTime);
          osc.stop(ctx.currentTime + 0.8);
          alertPlayedRef.current = true;
        } catch (_) {}
      }
      if (!data.drift_alert) alertPlayedRef.current = false;
    } catch (_) {
    } finally {
      setLoading(false);
    }
  }, [forecastVec, actualVec, actualBlocks, selectedDate, selectedRegion, audioEnabled]);

  useEffect(() => {
    fetchAccuracy();
    const id = setInterval(fetchAccuracy, REFRESH_MS);
    return () => clearInterval(id);
  }, [fetchAccuracy]);

  const handleRecalibrate = async () => {
    setRecalibrating(true);
    setRecalibMsg(null);
    try {
      await axios.post(getApiUrl('/api/v2/live'), {
        date: selectedDate,
        region: selectedRegion,
        actual_blocks: actualBlocks,
      });
      setRecalibMsg('Reforecast complete — refresh the Forecast tab to see updated values.');
      await fetchAccuracy();
    } catch (e) {
      setRecalibMsg(`Reforecast failed: ${e?.response?.data?.detail || e.message}`);
    } finally {
      setRecalibrating(false);
    }
  };

  // Per-block deviation rows
  const blockRows = useMemo(() => {
    return Array.from({ length: 96 }, (_, i) => {
      const block = i + 1;
      const isSettled = i < actualBlocks;
      const forecast = forecastDf[i]?.forecast_mw ?? forecastVec[i] ?? null;
      const actual = isSettled ? (forecastDf[i]?.actual_mw ?? actualVec[i] ?? null) : null;
      const ape =
        actual != null && actual > 10 && forecast != null && forecast > 10
          ? (Math.abs(actual - forecast) / actual) * 100
          : null;
      return { block, isSettled, forecast, actual, ape };
    });
  }, [forecastDf, forecastVec, actualVec, actualBlocks]);

  const currentMape = accuracyData?.current_mape;
  const driftAlert = accuracyData?.drift_alert;
  const recalibRecommended = accuracyData?.recalibrate_recommended;
  const worst3 = accuracyData?.worst_3_blocks || [];

  const statusColor = currentMape == null
    ? 'var(--text-muted)'
    : currentMape < 2 ? 'var(--success)'
    : currentMape < 4 ? 'color-mix(in srgb, var(--success) 55%, var(--warning))'
    : currentMape < 6 ? 'var(--warning)'
    : 'var(--danger)';

  const statusLabel = currentMape == null
    ? 'NO DATA'
    : currentMape < 2 ? 'ON TARGET'
    : currentMape < 4 ? 'TRACKING'
    : currentMape < 6 ? 'DRIFTING'
    : 'CRITICAL';

  return (
    <div className="am-page">
      {/* Header bar */}
      <div className="am-header">
        <div className="am-header__left">
          <span className="am-title">Accuracy Monitor</span>
          {selectedRegion && (
            <span className="am-region-badge">{selectedRegion.toUpperCase()}</span>
          )}
          {driftAlert && (
            <span className="am-drift-badge">⚠ DRIFT DETECTED</span>
          )}
        </div>
        <div className="am-header__right">
          <span className="am-mape-chip" style={{ color: statusColor, borderColor: statusColor + '40' }}>
            MAPE {currentMape != null ? `${currentMape.toFixed(2)}%` : '—'} · {statusLabel}
          </span>
          <span className="am-target-chip">Target &lt;2%</span>
          <button
            className="am-audio-btn"
            onClick={() => setAudioEnabled((v) => !v)}
            title={audioEnabled ? 'Mute drift alerts' : 'Enable drift alert sound'}
          >
            {audioEnabled ? '🔊' : '🔇'}
          </button>
          <button
            className="am-refresh-btn"
            onClick={fetchAccuracy}
            disabled={loading}
            title="Refresh accuracy"
          >
            {loading ? '...' : '↻ Refresh'}
          </button>
          {lastRefresh && (
            <span className="am-refresh-time">
              Updated {lastRefresh.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}
            </span>
          )}
        </div>
      </div>

      {/* Drift alert ribbon */}
      {driftAlert && (
        <div className="am-alert-ribbon">
          <span>
            ⚡ Forecast drift detected — {actualBlocks} settled blocks, running MAPE {currentMape?.toFixed(1)}%
          </span>
          {recalibRecommended && (
            <button
              className="am-recalib-btn"
              onClick={handleRecalibrate}
              disabled={recalibrating}
            >
              {recalibrating ? 'Recalibrating…' : '⚡ Trigger Intraday Reforecast'}
            </button>
          )}
        </div>
      )}
      {recalibMsg && (
        <div className="am-recalib-msg">{recalibMsg}</div>
      )}

      {/* KPI strip */}
      <div className="am-kpi-strip">
        <div className="am-kpi">
          <span className="am-kpi__label">Settled Blocks</span>
          <span className="am-kpi__value">{actualBlocks}<span className="am-kpi__unit">/96</span></span>
        </div>
        <div className="am-kpi">
          <span className="am-kpi__label">Running MAPE</span>
          <span className="am-kpi__value" style={{ color: statusColor }}>
            {currentMape != null ? `${currentMape.toFixed(2)}%` : '—'}
          </span>
        </div>
        <div className="am-kpi">
          <span className="am-kpi__label">Accuracy</span>
          <span className="am-kpi__value" style={{ color: statusColor }}>
            {currentMape != null ? `${(100 - currentMape).toFixed(2)}%` : '—'}
          </span>
        </div>
        <div className="am-kpi">
          <span className="am-kpi__label">Status</span>
          <span className="am-kpi__value" style={{ color: statusColor }}>{statusLabel}</span>
        </div>
        {worst3[0] && (
          <div className="am-kpi">
            <span className="am-kpi__label">Worst Block</span>
            <span className="am-kpi__value" style={{ color: 'var(--danger)' }}>
              B{worst3[0].block} ({worst3[0].ape?.toFixed(1)}%)
            </span>
          </div>
        )}
      </div>

      {/* 96-block grid */}
      <div className="am-section-title">96-Block Live Grid</div>
      <div className="am-grid">
        {blockRows.map(({ block, isSettled, forecast, actual, ape }) => (
          <div
            key={block}
            className="am-block"
            style={{ background: devBg(ape, isSettled) }}
            title={`B${block} ${blockTime(block)}${actual != null ? `\nActual: ${Math.round(actual)} MW` : ''}\nForecast: ${forecast != null ? Math.round(forecast) : '—'} MW${ape != null ? `\nAPE: ${ape.toFixed(1)}%` : ''}`}
          >
            <span className="am-block__num">{block}</span>
            <span
              className="am-block__dot"
              style={{ background: isSettled ? devColor(ape) : 'var(--outline-light)' }}
            />
            {ape != null && (
              <span className="am-block__ape" style={{ color: devColor(ape) }}>
                {ape.toFixed(1)}
              </span>
            )}
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="am-legend">
        <span>APE %:</span>
        {[['≤2%', 'var(--success)'], ['2–5%', 'color-mix(in srgb, var(--success) 55%, var(--warning))'], ['5–10%', 'var(--warning)'], ['>10%', 'var(--danger)']].map(([label, color]) => (
          <span key={label} className="am-legend__item">
            <span style={{ width: 10, height: 10, borderRadius: 2, background: color, display: 'inline-block', marginRight: 4 }} />
            {label}
          </span>
        ))}
        <span className="am-legend__item"><span style={{ width: 10, height: 10, borderRadius: 2, background: 'var(--outline-light)', display: 'inline-block', marginRight: 4 }} /> Forecast (unsettled)</span>
      </div>

      {/* Worst blocks table */}
      {worst3.length > 0 && (
        <>
          <div className="am-section-title" style={{ marginTop: 20 }}>Worst Performing Blocks</div>
          <div className="am-worst-table">
            <div className="am-wt-head">
              <span>Block</span><span>Time</span><span>Actual</span><span>Forecast</span><span>APE</span>
            </div>
            {worst3.map((w) => (
              <div key={w.block} className="am-wt-row">
                <span>B{w.block}</span>
                <span>{blockTime(w.block)}</span>
                <span>{w.actual_mw?.toFixed(0)} MW</span>
                <span>{w.forecast_mw?.toFixed(0)} MW</span>
                <span style={{ color: devColor(w.ape) }}>{w.ape?.toFixed(2)}%</span>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Intraday recalibration status */}
      {series.intraday_recalibrated && (
        <div className="am-recalib-status">
          ✓ Intraday recalibration applied at block {series.intraday_recalibration_blocks} — morning bias corrected
        </div>
      )}
    </div>
  );
}
