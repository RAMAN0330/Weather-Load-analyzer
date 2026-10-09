import React, { useCallback, useMemo, useState } from 'react';
import axios from 'axios';
import { getTrainingApiUrl } from '../../apiConfig';

const REGIONS = ['haryana', 'odisha', 'rajasthan', 'chhattisgarh'];

const fmtDate = (iso) => {
  try {
    return new Date(iso + 'T00:00:00').toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
  } catch {
    return iso;
  }
};

const mapeColor = (v) => {
  if (v == null) return 'var(--text-muted)';
  if (v < 2) return 'var(--success)';
  if (v < 4) return 'color-mix(in srgb, var(--success) 55%, var(--warning))';
  if (v < 6) return 'var(--warning)';
  return 'var(--danger)';
};

const today = () => new Date().toISOString().slice(0, 10);
const daysAgo = (n) => new Date(Date.now() - n * 86400000).toISOString().slice(0, 10);

export default function BacktestPage({ selectedRegion }) {
  const [dateFrom, setDateFrom] = useState(daysAgo(7));
  const [dateTo, setDateTo] = useState(today());
  const [region, setRegion] = useState(selectedRegion || 'haryana');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const runBacktest = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const { data } = await axios.post(getTrainingApiUrl('/v2/backtest'), {
        region,
        date_from: dateFrom,
        date_to: dateTo,
      }, { timeout: 300000 });
      setResults(data);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || 'Backtest failed');
    } finally {
      setLoading(false);
    }
  }, [region, dateFrom, dateTo]);

  const { daily, summary } = useMemo(() => {
    if (!results) return { daily: [], summary: null };
    const daily = results.daily_results || [];
    const t1Mapes = daily.map((d) => d.t1_mape).filter((v) => v != null);
    const t2Mapes = daily.map((d) => d.t2_mape).filter((v) => v != null);
    const avg = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
    const summary = {
      t1_avg_mape: avg(t1Mapes),
      t2_avg_mape: avg(t2Mapes),
      t1_accuracy: t1Mapes.length ? 100 - avg(t1Mapes) : null,
      t2_accuracy: t2Mapes.length ? 100 - avg(t2Mapes) : null,
      days: daily.length,
      t1_above_98: t1Mapes.filter((v) => 100 - v >= 98).length,
      t2_above_96: t2Mapes.filter((v) => 100 - v >= 96).length,
    };
    return { daily, summary };
  }, [results]);

  return (
    <div className="bt-page">
      <div className="bt-header">
        <span className="bt-title">Backtest — Accuracy Regression Detection</span>
      </div>

      {/* Controls */}
      <div className="bt-controls">
        <div className="bt-field">
          <label className="bt-label">Region</label>
          <select className="bt-select" value={region} onChange={(e) => setRegion(e.target.value)}>
            {REGIONS.map((r) => (
              <option key={r} value={r}>{r.charAt(0).toUpperCase() + r.slice(1)}</option>
            ))}
          </select>
        </div>
        <div className="bt-field">
          <label className="bt-label">Date From</label>
          <input type="date" className="bt-input" value={dateFrom} max={dateTo} onChange={(e) => setDateFrom(e.target.value)} />
        </div>
        <div className="bt-field">
          <label className="bt-label">Date To</label>
          <input type="date" className="bt-input" value={dateTo} min={dateFrom} max={today()} onChange={(e) => setDateTo(e.target.value)} />
        </div>
        <div className="bt-presets">
          {[[7, '7D'], [14, '14D']].map(([d, label]) => (
            <button key={d} className="bt-preset" onClick={() => { setDateFrom(daysAgo(d)); setDateTo(today()); }}>
              {label}
            </button>
          ))}
        </div>
        <button className="bt-run-btn" onClick={runBacktest} disabled={loading}>
          {loading ? 'Running…' : '▶ Run Backtest'}
        </button>
      </div>

      {error && <div className="bt-error">{error}</div>}

      {loading && (
        <div className="bt-loading">
          <div className="bt-spinner" />
          Running backtest…
        </div>
      )}

      {/* Summary */}
      {summary && (
        <div className="bt-summary">
          <div className="bt-sum-kpi">
            <span className="bt-sum-label">Days Analysed</span>
            <span className="bt-sum-value">{summary.days}</span>
          </div>
          <div className="bt-sum-kpi">
            <span className="bt-sum-label">T+1 Avg MAPE</span>
            <span className="bt-sum-value" style={{ color: mapeColor(summary.t1_avg_mape) }}>
              {summary.t1_avg_mape != null ? `${summary.t1_avg_mape.toFixed(2)}%` : '—'}
            </span>
          </div>
          <div className="bt-sum-kpi">
            <span className="bt-sum-label">T+1 Avg Accuracy</span>
            <span className="bt-sum-value" style={{ color: mapeColor(summary.t1_avg_mape) }}>
              {summary.t1_accuracy != null ? `${summary.t1_accuracy.toFixed(2)}%` : '—'}
            </span>
          </div>
          <div className="bt-sum-kpi">
            <span className="bt-sum-label">T+2 Avg MAPE</span>
            <span className="bt-sum-value" style={{ color: mapeColor(summary.t2_avg_mape) }}>
              {summary.t2_avg_mape != null ? `${summary.t2_avg_mape.toFixed(2)}%` : '—'}
            </span>
          </div>
          <div className="bt-sum-kpi">
            <span className="bt-sum-label">T+2 Avg Accuracy</span>
            <span className="bt-sum-value" style={{ color: mapeColor(summary.t2_avg_mape) }}>
              {summary.t2_accuracy != null ? `${summary.t2_accuracy.toFixed(2)}%` : '—'}
            </span>
          </div>
          <div className="bt-sum-kpi">
            <span className="bt-sum-label">Days ≥98% T+1</span>
            <span className="bt-sum-value" style={{ color: 'var(--success)' }}>
              {summary.t1_above_98}/{summary.days}
            </span>
          </div>
          <div className="bt-sum-kpi">
            <span className="bt-sum-label">Days ≥96% T+2</span>
            <span className="bt-sum-value" style={{ color: 'var(--success)' }}>
              {summary.t2_above_96}/{summary.days}
            </span>
          </div>
        </div>
      )}

      {/* Target gap */}
      {summary?.t1_avg_mape != null && (
        <div className="bt-target-gap" style={{ color: summary.t1_avg_mape < 2 ? 'var(--success)' : 'var(--warning)' }}>
          {summary.t1_avg_mape < 2
            ? '✓ T+1 target achieved (MAPE < 2%)'
            : `T+1 target gap: ${(summary.t1_avg_mape - 2).toFixed(2)}% above 2% target`}
        </div>
      )}

      {/* Daily results table */}
      {daily.length > 0 && (
        <>
          <div className="bt-section-title">Daily Results</div>
          <div className="bt-table-wrap">
            <div className="bt-table">
              <div className="bt-thead">
                <span>Date</span>
                <span>T+1 MAPE</span>
                <span>T+1 Acc</span>
                <span>T+2 MAPE</span>
                <span>T+2 Acc</span>
                <span>Worst Block</span>
                <span>Weather</span>
              </div>
              {daily.map((row) => (
                <div
                  key={row.date}
                  className={`bt-trow ${(row.t1_mape > 4 || row.t2_mape > 6) ? 'bt-trow--warn' : ''}`}
                >
                  <span>{fmtDate(row.date)}</span>
                  <span style={{ color: mapeColor(row.t1_mape) }}>
                    {row.t1_mape != null ? `${row.t1_mape.toFixed(2)}%` : '—'}
                  </span>
                  <span style={{ color: mapeColor(row.t1_mape) }}>
                    {row.t1_mape != null ? `${(100 - row.t1_mape).toFixed(2)}%` : '—'}
                  </span>
                  <span style={{ color: mapeColor(row.t2_mape) }}>
                    {row.t2_mape != null ? `${row.t2_mape.toFixed(2)}%` : '—'}
                  </span>
                  <span style={{ color: mapeColor(row.t2_mape) }}>
                    {row.t2_mape != null ? `${(100 - row.t2_mape).toFixed(2)}%` : '—'}
                  </span>
                  <span style={{ color: row.worst_block_ape > 10 ? 'var(--danger)' : 'var(--text-muted)' }}>
                    {row.worst_block != null ? `B${row.worst_block} (${row.worst_block_ape?.toFixed(1)}%)` : '—'}
                  </span>
                  <span className="bt-weather-flag">{row.weather_flag || '—'}</span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {/* MAPE bar chart */}
      {daily.length > 0 && (
        <>
          <div className="bt-section-title" style={{ marginTop: 20 }}>T+1 MAPE Trend</div>
          <div className="bt-chart">
            {daily.map((row, i) => (
              <div key={row.date} className="bt-bar-wrap" title={`${fmtDate(row.date)}: ${row.t1_mape?.toFixed(2)}%`}>
                <div
                  className="bt-bar"
                  style={{
                    height: row.t1_mape != null ? `${Math.min(100, row.t1_mape * 10)}%` : '2%',
                    background: mapeColor(row.t1_mape),
                  }}
                />
              </div>
            ))}
            {/* 2% target line */}
            <div className="bt-target-line" style={{ bottom: '20%' }} title="2% MAPE target" />
          </div>
          <div className="bt-chart-legend">
            <span style={{ color: 'var(--success)' }}>■ ≤2%</span>
            <span style={{ color: 'color-mix(in srgb, var(--success) 55%, var(--warning))' }}>■ 2–4%</span>
            <span style={{ color: 'var(--warning)' }}>■ 4–6%</span>
            <span style={{ color: 'var(--danger)' }}>■ &gt;6%</span>
            <span style={{ color: 'var(--text-muted)', marginLeft: 8 }}>— 2% target</span>
          </div>
        </>
      )}
    </div>
  );
}
