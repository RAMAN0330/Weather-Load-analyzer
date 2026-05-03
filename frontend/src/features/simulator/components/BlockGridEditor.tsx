import React, { useEffect, useMemo, useState } from 'react';
import { useSimulatorStore } from '../store';
import type { DriverType } from '../types';

type ColumnKey =
  | DriverType
  | 'net_pct'
  | 'temp_exog'
  | 'hum_exog'
  | 'precip_exog'
  | 'wind_exog'
  | 'weather_total_exog'
  | 'residual_delta'
  | 'weather_dir'
  | 'dominant_exog'
  | 'actual_vs_weather'
  | 'weight_conf'
  | 'regime_conf'
  | 'actual_mw';

const cols: Array<{ key: ColumnKey; label: string; editable?: boolean }> = [
  { key: 'weather_pct', label: 'Weather' },
  { key: 'daytype_pct', label: 'Day' },
  { key: 'holiday_pct', label: 'Holiday' },
  { key: 'manual_pct', label: 'Manual' },
  { key: 'weather_total_exog', label: 'Exog Total', editable: false },
  { key: 'temp_exog', label: 'Temp Exog', editable: false },
  { key: 'hum_exog', label: 'Hum Exog', editable: false },
  { key: 'precip_exog', label: 'Precip Exog', editable: false },
  { key: 'wind_exog', label: 'Wind Exog', editable: false },
  { key: 'residual_delta', label: 'Residual', editable: false },
  { key: 'weather_dir', label: 'Direction', editable: false },
  { key: 'dominant_exog', label: 'Top Exog', editable: false },
  { key: 'actual_vs_weather', label: 'Wx vs Actual', editable: false },
  { key: 'weight_conf', label: 'Weight Conf', editable: false },
  { key: 'regime_conf', label: 'Regime Conf', editable: false },
  { key: 'actual_mw', label: 'Actual MW', editable: false },
  { key: 'net_pct', label: 'Net', editable: false },
];

const isDriverKey = (key: ColumnKey): key is DriverType =>
  key === 'weather_pct' || key === 'daytype_pct' || key === 'holiday_pct' || key === 'manual_pct';

const formatSignedPercent = (value: number | null) => {
  if (value == null || !Number.isFinite(value)) return '--';
  return `${value >= 0 ? '+' : ''}${value.toFixed(3)}%`;
};

const formatPercent = (value: number | null) => {
  if (value == null || !Number.isFinite(value)) return '--';
  return `${value.toFixed(1)}%`;
};

const formatMw = (value: number | null) => {
  if (value == null || !Number.isFinite(value)) return '--';
  return `${value.toFixed(1)} MW`;
};

const titleCase = (value: string) =>
  String(value || '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (ch) => ch.toUpperCase());

export const BlockGridEditor: React.FC = () => {
  const blocks = useSimulatorStore((s) => s.blocks);
  const selectedBlocks = useSimulatorStore((s) => s.selectedBlocks);
  const updateDriver = useSimulatorStore((s) => s.updateDriver);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState<string>('');
  const hasScopedSelection = selectedBlocks.length > 0 && selectedBlocks.length < blocks.length;
  const [viewMode, setViewMode] = useState<'selection' | 'all'>(
    hasScopedSelection ? 'selection' : 'all'
  );

  useEffect(() => {
    if (!hasScopedSelection && viewMode === 'selection') {
      setViewMode('all');
    }
  }, [hasScopedSelection, viewMode]);

  const rowData = useMemo(() => {
    if (viewMode === 'selection' && hasScopedSelection) {
      const selectedSet = new Set(selectedBlocks);
      return blocks.filter((b) => selectedSet.has(b.block_number));
    }
    return blocks;
  }, [blocks, hasScopedSelection, selectedBlocks, viewMode]);

  const gridTemplateColumns = useMemo(() => `86px repeat(${cols.length}, minmax(110px, 1fr))`, []);

  const stats = useMemo(() => {
    if (!rowData.length) {
      return {
        avgNet: null,
        avgResidual: null,
        peakAdjusted: null,
        avgConfidence: null,
      };
    }
    const avgNet =
      rowData.reduce((sum, block) => sum + Number(block.net_pct || 0), 0) / rowData.length;
    const residualValues = rowData
      .map((block) => block.exog?.residual_delta_pct)
      .filter((value): value is number => value != null && Number.isFinite(Number(value)))
      .map((value) => Math.abs(Number(value)));
    const avgResidual = residualValues.length
      ? residualValues.reduce((sum, value) => sum + value, 0) / residualValues.length
      : null;
    const peakAdjusted = rowData.reduce<number | null>((peak, block) => {
      const next = Number(block.final_mw || 0);
      return peak == null || next > peak ? next : peak;
    }, null);
    const confidenceValues = rowData
      .flatMap((block) => [block.exog?.weight_confidence, block.exog?.regime_confidence])
      .filter((value): value is number => value != null && Number.isFinite(Number(value)))
      .map((value) => Number(value) * 100);
    const avgConfidence = confidenceValues.length
      ? confidenceValues.reduce((sum, value) => sum + value, 0) / confidenceValues.length
      : null;
    return {
      avgNet,
      avgResidual,
      peakAdjusted,
      avgConfidence,
    };
  }, [rowData]);

  const startEdit = (block: number, key: ColumnKey, current: number) => {
    if (!isDriverKey(key)) return;
    setEditing(`${block}:${key}`);
    setDraft(String(current.toFixed(3)));
  };

  const commitEdit = (block: number, key: ColumnKey) => {
    if (!isDriverKey(key)) return;
    const next = Number(draft);
    if (Number.isFinite(next)) {
      updateDriver(block, key, next);
    }
    setEditing(null);
    setDraft('');
  };

  return (
    <div className="sim-grid-wrap">
      <div className="sim-grid-toolbar">
        <div className="sim-grid-toolbar-copy">
          <strong>
            {viewMode === 'selection' && hasScopedSelection
              ? 'Focused block selection'
              : 'Full day block matrix'}
          </strong>
          <span>
            Editable driver cells stay upfront. Weather diagnostics, confidence, residuals, and
            actual load stay beside them for context.
          </span>
        </div>
        {hasScopedSelection ? (
          <div className="sim-grid-toggle">
            <button
              type="button"
              className={viewMode === 'selection' ? 'active' : ''}
              onClick={() => setViewMode('selection')}
            >
              Selected
            </button>
            <button
              type="button"
              className={viewMode === 'all' ? 'active' : ''}
              onClick={() => setViewMode('all')}
            >
              All 96
            </button>
          </div>
        ) : null}
      </div>

      <div className="sim-grid-stats">
        <div className="sim-grid-stat">
          <span className="sim-grid-stat-label">Rows in View</span>
          <strong>{rowData.length}</strong>
          <small>
            {viewMode === 'selection' && hasScopedSelection
              ? 'Focused edit scope'
              : 'Whole day coverage'}
          </small>
        </div>
        <div className="sim-grid-stat">
          <span className="sim-grid-stat-label">Avg Net Shift</span>
          <strong>{formatSignedPercent(stats.avgNet)}</strong>
          <small>Net driver movement across visible blocks</small>
        </div>
        <div className="sim-grid-stat">
          <span className="sim-grid-stat-label">Avg Residual</span>
          <strong>{formatPercent(stats.avgResidual)}</strong>
          <small>Absolute residual drift still unexplained</small>
        </div>
        <div className="sim-grid-stat">
          <span className="sim-grid-stat-label">Peak Adjusted</span>
          <strong>{formatMw(stats.peakAdjusted)}</strong>
          <small>Highest adjusted block inside this view</small>
        </div>
        <div className="sim-grid-stat">
          <span className="sim-grid-stat-label">Mean Confidence</span>
          <strong>{formatPercent(stats.avgConfidence)}</strong>
          <small>Average weight and regime confidence</small>
        </div>
      </div>

      <div className="sim-grid-scroll">
        <div className="sim-grid-head" style={{ gridTemplateColumns }}>
          <span className="sim-grid-index">Block</span>
          {cols.map((c) => (
            <span key={`h-${c.key}`}>{c.label}</span>
          ))}
        </div>
        {rowData.map((b) => (
          <div key={b.block_number} className="sim-grid-row" style={{ gridTemplateColumns }}>
            <span className="sim-grid-index">{b.block_number}</span>
            {cols.map((c) => {
              const key = c.key;
              const dominantExog = (() => {
                const temp = Number(b.exog?.temperature_pct || 0);
                const hum = Number(b.exog?.humidity_pct || 0);
                const precip = Number(b.exog?.precipitation_pct || 0);
                const wind = Number(b.exog?.wind_pct || 0);
                const maxAbs = Math.max(
                  Math.abs(temp),
                  Math.abs(hum),
                  Math.abs(precip),
                  Math.abs(wind)
                );
                if (maxAbs <= 1e-9) return 'neutral';
                if (Math.abs(temp) === maxAbs) return `temp ${temp >= 0 ? 'up' : 'down'}`;
                if (Math.abs(hum) === maxAbs) return `humidity ${hum >= 0 ? 'up' : 'down'}`;
                if (Math.abs(precip) === maxAbs) return `precip ${precip >= 0 ? 'up' : 'down'}`;
                return `wind ${wind >= 0 ? 'up' : 'down'}`;
              })();
              const value =
                key === 'net_pct'
                  ? b.net_pct
                  : key === 'weather_total_exog'
                    ? Number(b.exog?.weather_total_pct || 0)
                    : key === 'temp_exog'
                      ? Number(b.exog?.temperature_pct || 0)
                      : key === 'hum_exog'
                        ? Number(b.exog?.humidity_pct || 0)
                        : key === 'precip_exog'
                          ? Number(b.exog?.precipitation_pct || 0)
                          : key === 'wind_exog'
                            ? Number(b.exog?.wind_pct || 0)
                            : key === 'residual_delta'
                              ? b.exog?.residual_delta_pct == null ||
                                !Number.isFinite(Number(b.exog?.residual_delta_pct))
                                ? null
                                : Number(b.exog?.residual_delta_pct)
                              : key === 'dominant_exog'
                                ? dominantExog
                                : key === 'weather_dir'
                                  ? String(b.exog?.direction || 'neutral')
                                  : key === 'actual_vs_weather'
                                    ? String(b.exog?.actual_vs_weather_alignment || 'na')
                                    : key === 'weight_conf'
                                      ? b.exog?.weight_confidence == null ||
                                        !Number.isFinite(Number(b.exog?.weight_confidence))
                                        ? null
                                        : Number(b.exog?.weight_confidence) * 100
                                      : key === 'regime_conf'
                                        ? b.exog?.regime_confidence == null ||
                                          !Number.isFinite(Number(b.exog?.regime_confidence))
                                          ? null
                                          : Number(b.exog?.regime_confidence) * 100
                                        : key === 'actual_mw'
                                          ? b.actual_mw == null ||
                                            !Number.isFinite(Number(b.actual_mw))
                                            ? null
                                            : Number(b.actual_mw)
                                          : b.drivers[key];
              const cellId = `${b.block_number}:${key}`;
              const isEditing = editing === cellId;
              const isEditableDriver = c.editable !== false && isDriverKey(key);
              if (isEditing) {
                return (
                  <input
                    key={cellId}
                    className="sim-cell-input"
                    autoFocus
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onBlur={() => commitEdit(b.block_number, key)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') commitEdit(b.block_number, key);
                      if (e.key === 'Escape') setEditing(null);
                    }}
                  />
                );
              }
              const isPercentMetric =
                key !== 'actual_mw' &&
                key !== 'dominant_exog' &&
                key !== 'weather_dir' &&
                key !== 'actual_vs_weather';
              const text =
                key === 'actual_mw'
                  ? formatMw(value == null ? null : Number(value))
                  : key === 'weight_conf' || key === 'regime_conf'
                    ? formatPercent(value == null ? null : Number(value))
                    : typeof value === 'number'
                      ? isPercentMetric
                        ? formatSignedPercent(Number(value))
                        : String(value)
                      : titleCase(String(value));
              const toneClass =
                typeof value === 'number' && Number.isFinite(value)
                  ? value > 0
                    ? 'is-positive'
                    : value < 0
                      ? 'is-negative'
                      : 'is-neutral'
                  : 'is-neutral';

              if (!isEditableDriver) {
                return (
                  <span key={cellId} className={`sim-cell-readonly ${toneClass}`}>
                    {text}
                  </span>
                );
              }

              return (
                <button
                  key={cellId}
                  type="button"
                  className="sim-cell-btn is-editable"
                  onClick={() => startEdit(b.block_number, key, Number(value))}
                >
                  {text}
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
};
