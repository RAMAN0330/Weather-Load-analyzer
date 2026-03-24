import React, { useMemo, useState } from 'react';

const blockTime = (block) => {
  const minutes = (block - 1) * 15;
  const h = String(Math.floor(minutes / 60)).padStart(2, '0');
  const m = String(minutes % 60).padStart(2, '0');
  return `${h}:${m}`;
};

const ForecastTable = ({ liveData, forecastUncertainty, actualBlocks, fmt }) => {
  const [filter, setFilter] = useState('all'); // all | deviations | upcoming

  const rows = useMemo(() => {
    if (!liveData?.length) return [];
    return liveData.map((d, idx) => {
      const block = d.block_number;
      const actual = d.actual_mw;
      const forecast = d.forecast_mw;
      const baseline = d.baseline_mw;
      const deviation = actual != null && actual > 0 && forecast > 0
        ? ((actual - forecast) / Math.max(forecast, 1)) * 100
        : null;
      const unc = forecastUncertainty?.[idx];
      return {
        block,
        time: blockTime(block),
        actual,
        forecast,
        baseline,
        deviation,
        p10: unc?.p10_mw,
        p50: unc?.p50_mw,
        p90: unc?.p90_mw,
        confidence: unc?.forecast_confidence,
        isActual: idx < (actualBlocks || 0),
      };
    });
  }, [liveData, forecastUncertainty, actualBlocks]);

  const filtered = useMemo(() => {
    if (filter === 'deviations') return rows.filter(r => r.deviation != null && Math.abs(r.deviation) > 3);
    if (filter === 'upcoming') return rows.filter(r => !r.isActual);
    return rows;
  }, [rows, filter]);

  if (!rows.length) return <div className="cp-empty">No forecast data</div>;

  return (
    <div className="forecast-table-wrap">
      <div className="ft-filters">
        {['all', 'deviations', 'upcoming'].map(f => (
          <button
            key={f}
            className={`ft-filter-btn ${filter === f ? 'active' : ''}`}
            onClick={() => setFilter(f)}
          >
            {f === 'all' ? 'All 96' : f === 'deviations' ? 'Deviations' : 'Upcoming'}
          </button>
        ))}
        <span className="ft-count">{filtered.length} blocks</span>
      </div>
      <div className="forecast-table">
        <div className="ft-head">
          <span>Block</span>
          <span>Time</span>
          <span>Forecast</span>
          <span>Actual</span>
          <span>Dev %</span>
          <span>P10</span>
          <span>P90</span>
          <span>Conf</span>
        </div>
        <div className="ft-body">
          {filtered.map(r => {
            const devClass = r.deviation != null
              ? Math.abs(r.deviation) > 10 ? 'ft-critical'
              : Math.abs(r.deviation) > 5 ? 'ft-warning'
              : 'ft-ok'
              : '';
            return (
              <div key={r.block} className={`ft-row ${r.isActual ? 'ft-actual' : 'ft-forecast'}`}>
                <span>{r.block}</span>
                <span>{r.time}</span>
                <span>{fmt(r.forecast)}</span>
                <span className="ft-actual-val">{r.actual != null && r.actual > 0 ? fmt(r.actual) : '—'}</span>
                <span className={devClass}>
                  {r.deviation != null ? `${r.deviation > 0 ? '+' : ''}${r.deviation.toFixed(1)}%` : '—'}
                </span>
                <span className="ft-dim">{r.p10 != null ? fmt(r.p10) : '—'}</span>
                <span className="ft-dim">{r.p90 != null ? fmt(r.p90) : '—'}</span>
                <span className="ft-dim">
                  {r.confidence != null ? `${(r.confidence * 100).toFixed(0)}%` : '—'}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default ForecastTable;
