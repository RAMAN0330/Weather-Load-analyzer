import React, { useMemo, useRef, useState } from "react";
import {
  Brush,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useSimulatorStore } from "../store";
import type { ChartBlockPoint } from "../types";
import { blockTimeLabel } from "../utils";

const chartMargins = { top: 16, right: 20, left: 10, bottom: 12 };
const round3 = (v: number) => Math.round((Number(v) || 0) * 1000) / 1000;
const signedPct = (v: number) => `${v >= 0 ? "+" : ""}${round3(v).toFixed(3)}%`;
const signedPctMaybe = (v: number | null | undefined) => (v == null || !Number.isFinite(Number(v)) ? "--" : signedPct(Number(v)));
const pct01Maybe = (v: number | null | undefined) => (v == null || !Number.isFinite(Number(v)) ? "--" : `${(Number(v) * 100).toFixed(1)}%`);
const coeffMaybe = (v: number | null | undefined) => (v == null || !Number.isFinite(Number(v)) ? "--" : Number(v).toFixed(5));

type DotProps = {
  cx?: number;
  cy?: number;
  payload?: ChartBlockPoint;
};

const DraggableDot = ({ cx = 0, cy = 0, payload }: DotProps) => {
  const updateDriver = useSimulatorStore((s) => s.updateDriver);
  const setSelectedBlock = useSimulatorStore((s) => s.setSelectedBlock);
  const startRef = useRef<{ y: number; pct: number } | null>(null);

  const onDown = (e: React.MouseEvent<SVGCircleElement>) => {
    if (!payload) return;
    e.stopPropagation();
    setSelectedBlock(payload.block);
    startRef.current = { y: e.clientY, pct: payload.manualPct };
    const onMove = (ev: MouseEvent) => {
      if (!startRef.current || !payload) return;
      const deltaY = startRef.current.y - ev.clientY;
      const nextPct = startRef.current.pct + (deltaY * 0.25);
      updateDriver(payload.block, "manual_pct", nextPct);
    };
    const onUp = () => {
      startRef.current = null;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  return <circle cx={cx} cy={cy} r={3.5} fill="#c7655f" stroke="#2b2d31" strokeWidth={1} onMouseDown={onDown} />;
};

export const ForecastChart: React.FC = () => {
  const blocks = useSimulatorStore((s) => s.blocks);
  const selectedBlocks = useSimulatorStore((s) => s.selectedBlocks);
  const setSelectedBlock = useSimulatorStore((s) => s.setSelectedBlock);
  const setRangeSelection = useSimulatorStore((s) => s.setRangeSelection);
  const dataDate = useSimulatorStore((s) => s.dataDate);
  const weightsSource = useSimulatorStore((s) => s.weightsSource);
  const partialBias = useSimulatorStore((s) => s.partialDayBiasCorrection);

  const [draggingRange, setDraggingRange] = useState(false);
  const [anchorBlock, setAnchorBlock] = useState<number | null>(null);
  const plotRef = useRef<HTMLDivElement | null>(null);

  const data = useMemo<ChartBlockPoint[]>(() => {
    return blocks.map((b) => {
      const base = b.baseline_mw;
      const w = b.drivers.weather_pct / 100;
      const d = b.drivers.daytype_pct / 100;
      const h = b.drivers.holiday_pct / 100;
      const m = b.drivers.manual_pct / 100;
      const weatherLayer = base * (1 + w);
      const daytypeLayer = base * (1 + w + d);
      const holidayLayer = base * (1 + w + d + h);
      const manualLayer = base * (1 + Math.max(-0.4, Math.min(0.5, w + d + h + m)));
      const tempExog = Number(b.exog?.temperature_pct || 0);
      const humExog = Number(b.exog?.humidity_pct || 0);
      const precipExog = Number(b.exog?.precipitation_pct || 0);
      const windExog = Number(b.exog?.wind_pct || 0);
      const absTemp = Math.abs(tempExog);
      const absHum = Math.abs(humExog);
      const absPrecip = Math.abs(precipExog);
      const absWind = Math.abs(windExog);
      const actualDeltaPct = (b.exog?.actual_delta_pct == null || !Number.isFinite(Number(b.exog?.actual_delta_pct))) ? null : Number(b.exog?.actual_delta_pct);
      const weatherTotalExogPct = Number(b.exog?.weather_total_pct ?? b.drivers.weather_pct);
      const weatherIncreaseDeltaPct = Number(b.exog?.weather_increase_delta_pct ?? Math.max(weatherTotalExogPct, 0));
      const weatherReductionDeltaPct = Number(b.exog?.weather_reduction_delta_pct ?? Math.min(weatherTotalExogPct, 0));
      const tempIncreaseDeltaPct = Number(b.exog?.temperature_increase_delta_pct ?? Math.max(tempExog, 0));
      const tempReductionDeltaPct = Number(b.exog?.temperature_reduction_delta_pct ?? Math.min(tempExog, 0));
      const actualIncreaseDeltaPct = actualDeltaPct == null
        ? null
        : Number(b.exog?.actual_increase_delta_pct ?? Math.max(actualDeltaPct, 0));
      const actualReductionDeltaPct = actualDeltaPct == null
        ? null
        : Number(b.exog?.actual_reduction_delta_pct ?? Math.min(actualDeltaPct, 0));
      const dominantExog =
        absTemp >= absHum && absTemp >= absPrecip && absTemp >= absWind
          ? "temperature"
          : absHum >= absTemp && absHum >= absPrecip && absHum >= absWind
            ? "humidity"
            : absPrecip >= absTemp && absPrecip >= absHum && absPrecip >= absWind
              ? "precipitation"
              : absWind >= absTemp && absWind >= absHum && absWind >= absPrecip
                ? "wind"
                : "mixed";
      const residualDeltaPct = (b.exog?.residual_delta_pct == null || !Number.isFinite(Number(b.exog?.residual_delta_pct)))
        ? null
        : Number(b.exog?.residual_delta_pct);
      return {
        block: b.block_number,
        timeLabel: blockTimeLabel(b.block_number),
        baseline: base,
        adjusted: b.final_mw,
        actual: (b.actual_mw == null || !Number.isFinite(Number(b.actual_mw))) ? null : Number(b.actual_mw),
        weatherLayer,
        daytypeLayer,
        holidayLayer,
        manualLayer,
        weatherPct: b.drivers.weather_pct,
        daytypePct: b.drivers.daytype_pct,
        holidayPct: b.drivers.holiday_pct,
        manualPct: b.drivers.manual_pct,
        tempExogPct: tempExog,
        humExogPct: humExog,
        precipExogPct: precipExog,
        windExogPct: windExog,
        weatherTotalExogPct,
        weatherIncreaseDeltaPct,
        weatherReductionDeltaPct,
        tempIncreaseDeltaPct,
        tempReductionDeltaPct,
        actualIncreaseDeltaPct,
        actualReductionDeltaPct,
        tempIncreaseCoeff: (b.exog?.temperature_increase_coeff == null || !Number.isFinite(Number(b.exog?.temperature_increase_coeff))) ? null : Number(b.exog?.temperature_increase_coeff),
        tempReductionCoeff: (b.exog?.temperature_reduction_coeff == null || !Number.isFinite(Number(b.exog?.temperature_reduction_coeff))) ? null : Number(b.exog?.temperature_reduction_coeff),
        weatherDirection: b.exog?.direction || "neutral",
        dominantExog: (absTemp === absHum && absHum === absPrecip) ? "mixed" : dominantExog,
        actualDeltaPct,
        actualDirection: b.exog?.actual_direction || "neutral",
        residualDeltaPct,
        residualDirection: b.exog?.residual_direction || "neutral",
        residualContributor: String(b.exog?.residual_contributor || "non_weather"),
        actualWeatherAlignment: b.exog?.actual_vs_weather_alignment || "na",
        actualWeatherExplanation: String(b.exog?.actual_vs_weather_explanation || ""),
        weightConfidence: (b.exog?.weight_confidence == null || !Number.isFinite(Number(b.exog?.weight_confidence))) ? null : Number(b.exog?.weight_confidence),
        regimeConfidence: (b.exog?.regime_confidence == null || !Number.isFinite(Number(b.exog?.regime_confidence))) ? null : Number(b.exog?.regime_confidence),
      };
    });
  }, [blocks]);

  const resolveBlockFromMouse = (clientX: number) => {
    if (!plotRef.current) return null;
    const rect = plotRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, clientX - rect.left));
    const frac = x / Math.max(rect.width, 1);
    return Math.max(1, Math.min(96, Math.round(1 + (95 * frac))));
  };

  const onMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!e.shiftKey) return;
    const block = resolveBlockFromMouse(e.clientX);
    if (!block) return;
    setDraggingRange(true);
    setAnchorBlock(block);
    setRangeSelection(block, block);
  };

  const onMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!draggingRange || anchorBlock == null) return;
    const block = resolveBlockFromMouse(e.clientX);
    if (!block) return;
    setRangeSelection(anchorBlock, block);
  };

  const onMouseUp = () => {
    setDraggingRange(false);
    setAnchorBlock(null);
  };

  const dateInfo = useMemo(() => {
    const raw = String(dataDate || "").trim();
    if (!raw) return { season: "unknown", calendarDay: "unknown" };
    const dt = new Date(raw);
    if (Number.isNaN(dt.getTime())) return { season: "unknown", calendarDay: "unknown" };
    const m = dt.getMonth() + 1;
    const season =
      (m === 12 || m <= 2) ? "winter"
        : (m <= 5) ? "spring"
          : (m <= 8) ? "summer"
            : "fall";
    const dayName = dt.toLocaleDateString(undefined, { weekday: "long" });
    const dayType = (dt.getDay() === 0 || dt.getDay() === 6) ? "Weekend" : "Weekday";
    return { season, calendarDay: `${dayName} (${dayType})` };
  }, [dataDate]);

  const renderTooltip = ({ active, label, payload }: { active?: boolean; label?: number | string; payload?: Array<{ payload?: ChartBlockPoint }> }) => {
    if (!active || !payload?.length || !payload[0]?.payload) return null;
    const p = payload[0].payload;
    return (
      <div style={{ background: "#2b2d31", border: "1px solid rgba(240, 221, 199, 0.16)", borderRadius: 10, padding: "10px 12px", color: "#f0e7da" }}>
        <div style={{ fontWeight: 700, marginBottom: 4 }}>
          Block {label} ({blockTimeLabel(Number(label))})
        </div>
        <div style={{ fontSize: 12, opacity: 0.9, marginBottom: 6 }}>
          Season: {dateInfo.season} | Day: {dateInfo.calendarDay}
        </div>
        <div style={{ fontSize: 12 }}>Baseline: {p.baseline.toFixed(2)} MW</div>
        <div style={{ fontSize: 12, marginBottom: 6 }}>Final: {p.adjusted.toFixed(2)} MW</div>
        <div style={{ fontSize: 12, marginBottom: 6 }}>
          Actual: {Number.isFinite(Number(p.actual)) ? `${Number(p.actual).toFixed(2)} MW` : "--"}
        </div>
        <div style={{ fontSize: 12 }}>Actual Delta: {signedPctMaybe(p.actualDeltaPct)} ({p.actualDirection})</div>
        <div style={{ fontSize: 12 }}>Actual Increase Delta: {signedPctMaybe(p.actualIncreaseDeltaPct)}</div>
        <div style={{ fontSize: 12 }}>Actual Reduction Delta: {signedPctMaybe(p.actualReductionDeltaPct)}</div>
        <div style={{ fontSize: 12 }}>Weather Total: {signedPct(p.weatherTotalExogPct)} ({p.weatherDirection})</div>
        <div style={{ fontSize: 12 }}>Weather Increase Delta: {signedPct(p.weatherIncreaseDeltaPct)}</div>
        <div style={{ fontSize: 12 }}>Weather Reduction Delta: {signedPct(p.weatherReductionDeltaPct)}</div>
        <div style={{ fontSize: 12 }}>Residual (Actual-Weather): {signedPctMaybe(p.residualDeltaPct)} ({p.residualDirection})</div>
        <div style={{ fontSize: 12 }}>Weather vs Actual: {p.actualWeatherAlignment}</div>
        <div style={{ fontSize: 12 }}>Temp: {signedPct(p.tempExogPct)}</div>
        <div style={{ fontSize: 12 }}>Temp Increase Delta: {signedPct(p.tempIncreaseDeltaPct)}</div>
        <div style={{ fontSize: 12 }}>Temp Reduction Delta: {signedPct(p.tempReductionDeltaPct)}</div>
        <div style={{ fontSize: 12 }}>Temp Coeff (+/-): {coeffMaybe(p.tempIncreaseCoeff)} / {coeffMaybe(p.tempReductionCoeff)}</div>
        <div style={{ fontSize: 12 }}>Humidity: {signedPct(p.humExogPct)}</div>
        <div style={{ fontSize: 12 }}>Precipitation: {signedPct(p.precipExogPct)}</div>
        <div style={{ fontSize: 12 }}>Wind: {signedPct(p.windExogPct)}</div>
        <div style={{ fontSize: 12 }}>Weight Confidence: {pct01Maybe(p.weightConfidence)}</div>
        <div style={{ fontSize: 12 }}>Regime Confidence: {pct01Maybe(p.regimeConfidence)}</div>
        <div style={{ fontSize: 12, marginTop: 4 }}>Top Exog: {p.dominantExog}</div>
        <div style={{ fontSize: 12 }}>Model Source: {weightsSource || "unknown"}</div>
        <div style={{ fontSize: 12 }}>
          Bias Correction: {partialBias?.applied ? `Applied (${(partialBias?.avg_correction_pct ?? 0).toFixed(3)}%)` : "Not applied"}
        </div>
        {p.actualWeatherExplanation && (
          <div style={{ fontSize: 12, marginTop: 4, color: "#e1bd9f" }}>{p.actualWeatherExplanation}</div>
        )}
      </div>
    );
  };

  return (
    <div className="sim-chart-wrap" ref={plotRef} onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={onMouseUp}>
      <ResponsiveContainer width="100%" height={440}>
        <ComposedChart data={data} margin={chartMargins} onClick={(state) => {
          const b = state?.activeLabel;
          if (!b) return;
          setSelectedBlock(Number(b));
        }}>
          <CartesianGrid stroke="rgba(240, 221, 199, 0.14)" vertical={false} />
          <XAxis dataKey="block" tick={{ fill: "#b9ab96", fontSize: 11 }} />
          <YAxis tick={{ fill: "#b9ab96", fontSize: 11 }} width={56} />
          <Tooltip content={renderTooltip} />
          <Legend />
          <Brush dataKey="block" height={24} onChange={(range) => {
            const s = Number(range?.startIndex ?? 0) + 1;
            const e = Number(range?.endIndex ?? 0) + 1;
            if (s > 0 && e > 0) setRangeSelection(s, e);
          }} />

          <Line dataKey="baseline" name="Baseline Forecast" stroke="#c8b39b" strokeDasharray="5 5" dot={false} strokeWidth={2} />
          <Line dataKey="actual" name="Actual" stroke="#d2a46f" dot={false} strokeWidth={2} connectNulls={false} />
          <Line
            dataKey="adjusted"
            name="Adjusted Forecast"
            stroke="#4f7d5c"
            dot={(props: any) => {
              const { key, ...rest } = props;
              return <DraggableDot key={key} {...rest} />;
            }}
            strokeWidth={2.5}
          />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="sim-selection-note">
        {selectedBlocks.length > 1
          ? `Selected blocks: ${selectedBlocks[0]} to ${selectedBlocks[selectedBlocks.length - 1]} (${selectedBlocks.length})`
          : `Selected block: ${selectedBlocks[0] ?? "--"}`}
      </div>
    </div>
  );
};
