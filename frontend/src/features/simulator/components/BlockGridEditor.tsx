import React, { useMemo, useState } from "react";
import { useSimulatorStore } from "../store";
import type { DriverType } from "../types";

type ColumnKey =
  | DriverType
  | "net_pct"
  | "temp_exog"
  | "hum_exog"
  | "precip_exog"
  | "wind_exog"
  | "weather_total_exog"
  | "residual_delta"
  | "weather_dir"
  | "dominant_exog"
  | "actual_vs_weather"
  | "weight_conf"
  | "regime_conf"
  | "actual_mw";

const cols: Array<{ key: ColumnKey; label: string; editable?: boolean }> = [
  { key: "weather_pct", label: "Weather" },
  { key: "daytype_pct", label: "Day" },
  { key: "holiday_pct", label: "Holiday" },
  { key: "manual_pct", label: "Manual" },
  { key: "weather_total_exog", label: "Exog Total", editable: false },
  { key: "temp_exog", label: "Temp Exog", editable: false },
  { key: "hum_exog", label: "Hum Exog", editable: false },
  { key: "precip_exog", label: "Precip Exog", editable: false },
  { key: "wind_exog", label: "Wind Exog", editable: false },
  { key: "residual_delta", label: "Residual", editable: false },
  { key: "weather_dir", label: "Direction", editable: false },
  { key: "dominant_exog", label: "Top Exog", editable: false },
  { key: "actual_vs_weather", label: "Wx vs Actual", editable: false },
  { key: "weight_conf", label: "Weight Conf", editable: false },
  { key: "regime_conf", label: "Regime Conf", editable: false },
  { key: "actual_mw", label: "Actual MW", editable: false },
  { key: "net_pct", label: "Net", editable: false },
];

const isDriverKey = (key: ColumnKey): key is DriverType =>
  key === "weather_pct" || key === "daytype_pct" || key === "holiday_pct" || key === "manual_pct";

export const BlockGridEditor: React.FC = () => {
  const blocks = useSimulatorStore((s) => s.blocks);
  const updateDriver = useSimulatorStore((s) => s.updateDriver);
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState<string>("");

  const rowData = useMemo(() => blocks, [blocks]);
  const gridTemplateColumns = useMemo(() => `52px repeat(${cols.length}, minmax(0, 1fr))`, []);

  const startEdit = (block: number, key: ColumnKey, current: number) => {
    if (!isDriverKey(key)) return;
    setEditing(`${block}:${key}`);
    setDraft(String(current.toFixed(3)));
  };

  const commitEdit = (block: number, key: ColumnKey) => {
    if (!isDriverKey(key)) return;
    updateDriver(block, key, Number(draft));
    setEditing(null);
    setDraft("");
  };

  return (
    <div className="sim-grid-wrap">
      <div className="sim-grid-head" style={{ gridTemplateColumns }}>
        <span>Block</span>
        {cols.map((c) => (
          <span key={`h-${c.key}`}>{c.label}</span>
        ))}
      </div>
      <div className="sim-grid-body">
        {rowData.map((b) => (
          <div key={b.block_number} className="sim-grid-row" style={{ gridTemplateColumns }}>
            <span>{b.block_number}</span>
            {cols.map((c) => {
              const key = c.key;
              const dominantExog = (() => {
                const temp = Number(b.exog?.temperature_pct || 0);
                const hum = Number(b.exog?.humidity_pct || 0);
                const precip = Number(b.exog?.precipitation_pct || 0);
                const wind = Number(b.exog?.wind_pct || 0);
                const maxAbs = Math.max(Math.abs(temp), Math.abs(hum), Math.abs(precip), Math.abs(wind));
                if (maxAbs <= 1e-9) return "neutral";
                if (Math.abs(temp) === maxAbs) return `temp ${temp >= 0 ? "up" : "down"}`;
                if (Math.abs(hum) === maxAbs) return `humidity ${hum >= 0 ? "up" : "down"}`;
                if (Math.abs(precip) === maxAbs) return `precip ${precip >= 0 ? "up" : "down"}`;
                return `wind ${wind >= 0 ? "up" : "down"}`;
              })();
              const value =
                key === "net_pct"
                  ? b.net_pct
                  : key === "weather_total_exog"
                    ? Number(b.exog?.weather_total_pct || 0)
                  : key === "temp_exog"
                    ? Number(b.exog?.temperature_pct || 0)
                    : key === "hum_exog"
                      ? Number(b.exog?.humidity_pct || 0)
                      : key === "precip_exog"
                        ? Number(b.exog?.precipitation_pct || 0)
                        : key === "wind_exog"
                          ? Number(b.exog?.wind_pct || 0)
                          : key === "residual_delta"
                            ? ((b.exog?.residual_delta_pct == null || !Number.isFinite(Number(b.exog?.residual_delta_pct))) ? null : Number(b.exog?.residual_delta_pct))
                        : key === "dominant_exog"
                          ? dominantExog
                        : key === "weather_dir"
                          ? String(b.exog?.direction || "neutral")
                        : key === "actual_vs_weather"
                          ? String(b.exog?.actual_vs_weather_alignment || "na")
                        : key === "weight_conf"
                          ? ((b.exog?.weight_confidence == null || !Number.isFinite(Number(b.exog?.weight_confidence))) ? null : Number(b.exog?.weight_confidence) * 100)
                          : key === "regime_conf"
                            ? ((b.exog?.regime_confidence == null || !Number.isFinite(Number(b.exog?.regime_confidence))) ? null : Number(b.exog?.regime_confidence) * 100)
                        : key === "actual_mw"
                          ? (b.actual_mw == null || !Number.isFinite(Number(b.actual_mw)) ? null : Number(b.actual_mw))
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
                      if (e.key === "Enter") commitEdit(b.block_number, key);
                      if (e.key === "Escape") setEditing(null);
                    }}
                  />
                );
              }
              const text = key === "actual_mw"
                ? (value == null ? "--" : `${Number(value).toFixed(2)} MW`)
                : key === "weight_conf" || key === "regime_conf"
                  ? (value == null ? "--" : `${Number(value).toFixed(1)}%`)
                : (typeof value === "number" ? `${value.toFixed(3)}%` : value);
              return (
                <button
                  key={cellId}
                  type="button"
                  className="sim-cell-btn"
                  onClick={() => startEdit(b.block_number, key, Number(value))}
                  disabled={!isEditableDriver}
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
