import React, { useMemo, useState, useRef } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';

const ROW_HEIGHT = 34; // px — fixed height for every data row

const blockTime = (block) => {
  const minutes = (block - 1) * 15;
  const h = String(Math.floor(minutes / 60)).padStart(2, '0');
  const m = String(minutes % 60).padStart(2, '0');
  return `${h}:${m}`;
};

const COLS = [
  { key: 'block', label: 'Block', width: 52 },
  { key: 'time',  label: 'Time',  width: 56 },
  { key: 'forecast', label: 'Forecast', width: 80 },
  { key: 'actual',   label: 'Actual',   width: 80 },
  { key: 'deviation',label: 'Dev %',    width: 68 },
  { key: 'p10',      label: 'P10',      width: 72 },
  { key: 'p90',      label: 'P90',      width: 72 },
  { key: 'confidence', label: 'Conf',   width: 58 },
];

const ForecastTable = ({ liveData, forecastUncertainty, actualBlocks, fmt }) => {
  const [filter, setFilter] = useState('all');
  const scrollRef = useRef(null);

  const rows = useMemo(() => {
    if (!liveData?.length) return [];
    return liveData.map((d, idx) => {
      const block    = d.block_number;
      const actual   = d.actual_mw;
      const forecast = d.forecast_mw;
      const baseline = d.baseline_mw;
      const deviation =
        actual != null && actual > 0 && forecast > 0
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
    if (filter === 'upcoming')   return rows.filter(r => !r.isActual);
    return rows;
  }, [rows, filter]);

  const virtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 8,
  });

  if (!rows.length) return <div className="cp-empty">No forecast data</div>;

  const totalHeight = virtualizer.getTotalSize();
  const virtualItems = virtualizer.getVirtualItems();

  return (
    <div className="forecast-table-wrap">
      {/* Filter bar */}
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

      {/* Table shell */}
      <div className="forecast-table">
        {/* Sticky header */}
        <div className="ft-head">
          {COLS.map(c => (
            <span key={c.key} style={{ width: c.width, flexShrink: 0 }}>{c.label}</span>
          ))}
        </div>

        {/* Virtualized body */}
        <div
          ref={scrollRef}
          className="ft-body"
          style={{ height: Math.min(totalHeight, 420), overflowY: 'auto', position: 'relative' }}
        >
          {/* Total spacer so scrollbar reflects full content */}
          <div style={{ height: totalHeight, width: '100%', position: 'relative' }}>
            {virtualItems.map(vItem => {
              const r = filtered[vItem.index];
              const devClass =
                r.deviation != null
                  ? Math.abs(r.deviation) > 10 ? 'ft-critical'
                  : Math.abs(r.deviation) > 5  ? 'ft-warning'
                  : 'ft-ok'
                  : '';
              return (
                <div
                  key={r.block}
                  data-index={vItem.index}
                  ref={virtualizer.measureElement}
                  className={`ft-row ${r.isActual ? 'ft-actual' : 'ft-forecast'}`}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: ROW_HEIGHT,
                    transform: `translateY(${vItem.start}px)`,
                    display: 'flex',
                    alignItems: 'center',
                  }}
                >
                  <span style={{ width: COLS[0].width, flexShrink: 0 }}>{r.block}</span>
                  <span style={{ width: COLS[1].width, flexShrink: 0 }}>{r.time}</span>
                  <span style={{ width: COLS[2].width, flexShrink: 0 }}>{fmt(r.forecast)}</span>
                  <span style={{ width: COLS[3].width, flexShrink: 0 }} className="ft-actual-val">
                    {r.actual != null && r.actual > 0 ? fmt(r.actual) : '—'}
                  </span>
                  <span style={{ width: COLS[4].width, flexShrink: 0 }} className={devClass}>
                    {r.deviation != null
                      ? `${r.deviation > 0 ? '+' : ''}${r.deviation.toFixed(1)}%`
                      : '—'}
                  </span>
                  <span style={{ width: COLS[5].width, flexShrink: 0 }} className="ft-dim">
                    {r.p10 != null ? fmt(r.p10) : '—'}
                  </span>
                  <span style={{ width: COLS[6].width, flexShrink: 0 }} className="ft-dim">
                    {r.p90 != null ? fmt(r.p90) : '—'}
                  </span>
                  <span style={{ width: COLS[7].width, flexShrink: 0 }} className="ft-dim">
                    {r.confidence != null ? `${(r.confidence * 100).toFixed(0)}%` : '—'}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ForecastTable;
