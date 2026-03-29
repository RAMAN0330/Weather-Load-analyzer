import React, { useEffect, useMemo, useRef, useState } from "react";
import { useSimulatorStore, type WeatherComponentSliders } from "../store";
import { calculateWeatherImpact, type WeatherImpactSeason } from "../weatherImpactService";
import { loadSimulatorDateOptionsApi } from "../simulatorDataService";
import { blockTimeLabel } from "../utils";
import { Activity, Zap, ChevronDown, ChevronUp, Sliders } from "lucide-react";

const SLIDER_RANGE_MIN = -4;
const SLIDER_RANGE_MAX = 4;
const SLIDER_STEP = 0.01;

const WEATHER_COMPONENT_CONFIG: Array<{ key: "temperature" | "humidity" | "precipitation" | "wind"; label: string; min: number; max: number; step: number }> = [
  { key: "temperature", label: "Temperature", min: SLIDER_RANGE_MIN, max: SLIDER_RANGE_MAX, step: SLIDER_STEP },
  { key: "humidity", label: "Humidity", min: SLIDER_RANGE_MIN, max: SLIDER_RANGE_MAX, step: SLIDER_STEP },
  { key: "precipitation", label: "Precipitation", min: SLIDER_RANGE_MIN, max: SLIDER_RANGE_MAX, step: SLIDER_STEP },
  { key: "wind", label: "Wind Speed", min: SLIDER_RANGE_MIN, max: SLIDER_RANGE_MAX, step: SLIDER_STEP },
];

const ADVANCED_DRIVERS = [
  { key: "cloud_cover", label: "Cloud Cover", min: SLIDER_RANGE_MIN, max: SLIDER_RANGE_MAX, step: SLIDER_STEP },
  { key: "solar_irradiance", label: "Solar Intensity", min: SLIDER_RANGE_MIN, max: SLIDER_RANGE_MAX, step: SLIDER_STEP },
  { key: "pressure", label: "Barometric Pressure", min: SLIDER_RANGE_MIN, max: SLIDER_RANGE_MAX, step: SLIDER_STEP },
];

export type DriverEditorPanelPart = "selection" | "sliders" | "scenarios" | "adjustments" | "command" | "insights" | "attribution" | "patterns";

type DriverEditorPanelProps = {
  activeTab: "simulation" | "impact";
  part?: DriverEditorPanelPart;
  calendarPanelOpen?: boolean;
  onCalendarPanelOpenChange?: (open: boolean) => void;
  selectedDate?: string;
  calendarContextLabel?: string;
};

export const DriverEditorPanel: React.FC<DriverEditorPanelProps> = ({
  activeTab,
  part,
  calendarPanelOpen: controlledCalendarPanelOpen,
  onCalendarPanelOpenChange,
  selectedDate: propSelectedDate,
  calendarContextLabel,
}) => {
  const round3 = (v: number) => Math.round((Number(v) || 0) * 1000) / 1000;
  const blockEndLabel = (block: number) => {
    if (block >= 96) return "24:00";
    return blockTimeLabel(block + 1);
  };
  const toBlockRanges = (blocksInWindow: number[]) => {
    const sorted = Array.from(new Set((blocksInWindow || []).map((v) => Number(v)).filter((v) => Number.isFinite(v) && v >= 1 && v <= 96))).sort((a, b) => a - b);
    if (!sorted.length) return [];
    const ranges: Array<{ start: number; end: number; timeLabel: string; blockLabel: string }> = [];
    let start = sorted[0];
    let prev = sorted[0];
    for (let i = 1; i < sorted.length; i += 1) {
      const curr = sorted[i];
      if (curr === prev + 1) {
        prev = curr;
        continue;
      }
      ranges.push({
        start,
        end: prev,
        timeLabel: `${blockTimeLabel(start)}-${blockEndLabel(prev)}`,
        blockLabel: `B${start}${start === prev ? "" : `-${prev}`}`,
      });
      start = curr;
      prev = curr;
    }
    ranges.push({
      start,
      end: prev,
      timeLabel: `${blockTimeLabel(start)}-${blockEndLabel(prev)}`,
      blockLabel: `B${start}${start === prev ? "" : `-${prev}`}`,
    });
    return ranges;
  };
  const formatBlockWindows = (blocksInWindow: number[]) => {
    const ranges = toBlockRanges(blocksInWindow);
    if (!ranges.length) return "No slots detected";
    const windows = ranges.map((r) => `${r.timeLabel} (${r.blockLabel})`);
    const show = windows.slice(0, 3);
    return windows.length > 3 ? `${show.join(" | ")} | +${windows.length - 3} more` : show.join(" | ");
  };

  const blocks = useSimulatorStore((s) => s.blocks);
  const selectedBlocks = useSimulatorStore((s) => s.selectedBlocks);
  const setSelectedBlocks = useSimulatorStore((s) => s.setSelectedBlocks);
  const setRangeSelection = useSimulatorStore((s) => s.setRangeSelection);
  const setSelectedBlock = useSimulatorStore((s) => s.setSelectedBlock);
  const recalculateForecast = useSimulatorStore((s) => s.recalculateForecast);
  const pendingRecompute = useSimulatorStore((s) => s.pendingRecompute);
  const weatherComponentSliders = useSimulatorStore((s) => s.weatherComponentSliders);
  const setWeatherComponentSlider = useSimulatorStore((s) => s.setWeatherComponentSlider);
  const dataDate = useSimulatorStore((s) => s.dataDate);

  const [commandBusy, setCommandBusy] = useState(false);
  const [selectionMode, setSelectionMode] = useState<"all" | "range" | "single">("single");
  const [rangeStart, setRangeStart] = useState(60);
  const [rangeEnd, setRangeEnd] = useState(72);
  const [singleBlock, setSingleBlock] = useState(72);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [calendarPanelOpenLocal, setCalendarPanelOpenLocal] = useState(false);
  const calendarPanelOpen = controlledCalendarPanelOpen ?? calendarPanelOpenLocal;
  const setCalendarPanelOpen = (next: boolean) => {
    if (controlledCalendarPanelOpen === undefined) setCalendarPanelOpenLocal(next);
    onCalendarPanelOpenChange?.(next);
  };

  const selected = useMemo(() => blocks.filter((b) => selectedBlocks.includes(b.block_number)), [blocks, selectedBlocks]);
  const totalBlocks = blocks.length || 96;
  const selectionRanges = useMemo(() => toBlockRanges(selectedBlocks), [selectedBlocks]);
  const inferredSelectionMode = useMemo<"all" | "range" | "single">(() => {
    if (selectedBlocks.length >= totalBlocks) return "all";
    if (selectedBlocks.length <= 1) return "single";
    return "range";
  }, [selectedBlocks, totalBlocks]);

  const selectedDate = propSelectedDate || dataDate || "";

  useEffect(() => {
    setSelectionMode(inferredSelectionMode);
    if (!selectedBlocks.length) return;
    if (selectedBlocks.length === 1) {
      setSingleBlock(selectedBlocks[0]);
      setRangeStart(selectedBlocks[0]);
      setRangeEnd(selectedBlocks[0]);
      return;
    }
    const sorted = [...selectedBlocks].sort((a, b) => a - b);
    setRangeStart(sorted[0]);
    setRangeEnd(sorted[sorted.length - 1]);
    setSingleBlock(sorted[0]);
  }, [inferredSelectionMode, selectedBlocks]);

  const handleRunCommand = async () => {
    setCommandBusy(true);
    try {
      await recalculateForecast(selectedBlocks);
    } finally {
      setCommandBusy(false);
    }
  };

  const exogInsight = useMemo(() => {
    if (!selected.length) return { weatherTotal: 0, weightConfidence: 0 };
    const n = Math.max(1, selected.length);
    const weatherTotal = selected.reduce((s, b) => s + Number(b.exog?.weather_total_pct ?? b.drivers.weather_pct ?? 0), 0) / n;
    const weightConfidence = selected.reduce((s, b) => s + Number(b.exog?.weight_confidence || 0), 0) / n;
    return { weatherTotal: round3(weatherTotal), weightConfidence: round3(weightConfidence) };
  }, [selected]);

  const attributionSummary = useMemo(() => {
    let weather = 0;
    let calendar = 0;
    let manual = 0;
    const scopeBlocks = selected.length ? selected : blocks;
    scopeBlocks.forEach((b) => {
      const base = Number(b.baseline_mw || 0);
      weather += base * (Number(b.drivers.weather_pct || 0) / 100);
      calendar += base * ((Number(b.drivers.daytype_pct || 0) + Number(b.drivers.holiday_pct || 0)) / 100);
      manual += base * (Number(b.drivers.manual_pct || 0) / 100);
    });
    return { weather, calendar, manual };
  }, [selected, blocks]);

  const getPercentage = (val: number, min: number, max: number) => {
    return ((val - min) / (max - min)) * 100;
  };

  if (part === "selection") {
    const selectionSummary = inferredSelectionMode === "all"
      ? "Whole-day scope across all 96 blocks"
      : inferredSelectionMode === "single"
        ? `Focused on block ${singleBlock} • ${blockTimeLabel(singleBlock)}-${blockEndLabel(singleBlock)}`
        : formatBlockWindows(selectedBlocks);

    return (
      <div className="glass-panel p-4 flex flex-col gap-4">
        <div
          className="flex justify-between items-center bg-white/5 p-3 rounded-xl border border-[var(--outline)] cursor-pointer hover:bg-white/10 transition-colors"
          onClick={() => setCalendarPanelOpen(true)}
        >
          <div className="flex flex-col">
            <span className="text-[10px] uppercase text-[var(--muted)]">Reference Date</span>
            <span className="text-sm font-bold text-[var(--accent)]">{selectedDate || "--"}</span>
          </div>
          <span className="text-xs text-[var(--muted)]">{calendarContextLabel || "--"}</span>
        </div>

        <div className="flex flex-col gap-2">
          <div className="flex bg-black/20 p-1 rounded-lg border border-[var(--outline)]">
            {(["all", "range", "single"] as const).map(m => (
              <button
                type="button"
                key={m}
                className={`flex-1 py-1.5 text-[10px] font-bold uppercase rounded-md transition-all ${selectionMode === m ? 'bg-[var(--accent)] text-black shadow-lg shadow-[var(--accent)]/20' : 'text-[var(--muted)]'}`}
                onClick={() => {
                  setSelectionMode(m);
                  if (m === "all") setSelectedBlocks(Array.from({ length: 96 }, (_, i) => i + 1));
                  else if (m === "range") setRangeSelection(rangeStart, rangeEnd);
                  else setSelectedBlock(singleBlock);
                }}
              >
                {m}
              </button>
            ))}
          </div>

          <div className="rounded-lg border border-[var(--outline)] bg-white/[0.03] px-3 py-2">
            <div className="flex items-center justify-between gap-3">
              <span className="text-[10px] uppercase tracking-wide text-[var(--muted)]">Current Scope</span>
              <span className="text-[10px] font-bold text-[var(--accent)]">{selectedBlocks.length || totalBlocks} Blocks</span>
            </div>
            <div className="mt-1 text-[11px] text-[var(--text-secondary)] leading-5">
              {selectionSummary}
            </div>
            {selectionRanges.length > 1 ? (
              <div className="mt-2 text-[10px] text-[var(--muted)]">
                Multi-window selection: {selectionRanges.map((range) => range.blockLabel).join(", ")}
              </div>
            ) : null}
          </div>

          {selectionMode === "range" && (
            <div className="grid grid-cols-2 gap-2 mt-1">
              <div className="flex flex-col gap-1">
                <label className="text-[10px] uppercase text-[var(--muted)]">Start</label>
                <input
                  type="number"
                  className="bg-white/5 border border-[var(--outline)] rounded-lg p-2 text-xs text-white"
                  min={1} max={96} value={rangeStart}
                  onChange={e => {
                    const val = Math.max(1, Math.min(96, Number(e.target.value)));
                    setRangeStart(val);
                    setRangeSelection(val, rangeEnd);
                  }}
                />
              </div>
              <div className="flex flex-col gap-1">
                <label className="text-[10px] uppercase text-[var(--muted)]">End</label>
                <input
                  type="number"
                  className="bg-white/5 border border-[var(--outline)] rounded-lg p-2 text-xs text-white"
                  min={1} max={96} value={rangeEnd}
                  onChange={e => {
                    const val = Math.max(1, Math.min(96, Number(e.target.value)));
                    setRangeEnd(val);
                    setRangeSelection(rangeStart, val);
                  }}
                />
              </div>
            </div>
          )}

          {selectionMode === "single" && (
            <div className="flex flex-col gap-1 mt-1">
              <label className="text-[10px] uppercase text-[var(--muted)]">Block Number</label>
              <input
                type="number"
                className="bg-white/5 border border-[var(--outline)] rounded-lg p-2 text-xs text-white"
                min={1} max={96} value={singleBlock}
                onChange={e => {
                  const val = Math.max(1, Math.min(96, Number(e.target.value)));
                  setSingleBlock(val);
                  setSelectedBlock(val);
                }}
              />
            </div>
          )}
        </div>
      </div>
    );
  }

  // Split parts: "scenarios" = quick presets, "adjustments" = manual sliders
  if (part === "scenarios") {
    const SCENARIO_PRESETS = [
      { name: "Heatwave (+5°C)", icon: "🔥", temp: 2.5, humidity: -0.5, precipitation: 0, wind: -0.3, desc: "Extreme heat scenario" },
      { name: "Monsoon Onset", icon: "🌧️", temp: -1.0, humidity: 2.0, precipitation: 3.0, wind: 1.0, desc: "Heavy rain + cooling" },
      { name: "Cold Wave (-5°C)", icon: "❄️", temp: -2.5, humidity: 0.5, precipitation: 0, wind: 0.5, desc: "Winter heating demand" },
      { name: "Clear & Calm", icon: "☀️", temp: 0, humidity: 0, precipitation: 0, wind: 0, desc: "Reset to baseline" },
      { name: "Industrial Shutdown", icon: "🏭", temp: 0, humidity: 0, precipitation: 0, wind: 0, desc: "30% industrial reduction" },
    ];
    const applyScenario = (preset: typeof SCENARIO_PRESETS[0]) => {
      setWeatherComponentSlider("temperature", preset.temp, selectedBlocks);
      setWeatherComponentSlider("humidity", preset.humidity, selectedBlocks);
      setWeatherComponentSlider("precipitation", preset.precipitation, selectedBlocks);
      setWeatherComponentSlider("wind", preset.wind, selectedBlocks);
    };
    return (
      <div className="glass-panel p-4">
        <div className="grid grid-cols-2 gap-2">
          {SCENARIO_PRESETS.map((preset) => (
            <button
              key={preset.name}
              className="p-2 rounded-lg bg-white/5 border border-[var(--outline)] hover:bg-white/10 hover:border-[var(--accent)] transition-all text-left"
              onClick={() => applyScenario(preset)}
              title={preset.desc}
            >
              <div className="text-xs font-medium">{preset.icon} {preset.name}</div>
              <div className="text-[9px] text-[var(--muted)]">{preset.desc}</div>
            </button>
          ))}
        </div>
      </div>
    );
  }

  if (part === "adjustments") {
    const ICONS: Record<string, string> = { temperature: '🌡️', humidity: '💧', precipitation: '🌧️', wind: '💨' };
    const ADV_ICONS: Record<string, string> = { cloud_cover: '☁️', solar_irradiance: '☀️', pressure: '🔵' };
    const allDrivers = showAdvanced ? [...WEATHER_COMPONENT_CONFIG, ...ADVANCED_DRIVERS] : WEATHER_COMPONENT_CONFIG;
    return (
      <div className="glass-panel p-3 flex flex-col gap-2">
        {allDrivers.map((driver) => {
          const val = Number(weatherComponentSliders[driver.key as keyof WeatherComponentSliders] || 0);
          const percent = getPercentage(val, driver.min, driver.max);
          const icon = { ...ICONS, ...ADV_ICONS }[driver.key] || '';
          const isActive = Math.abs(val) > 0.01;
          return (
            <div key={driver.key} className="rounded-lg border border-[var(--outline)] p-2" style={{ background: isActive ? 'rgba(255,255,255,0.03)' : 'transparent' }}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[11px] text-[var(--muted)]">{icon} {driver.label}</span>
                <span className={`text-xs font-mono font-bold ${isActive ? 'text-[var(--accent)]' : 'text-[var(--muted)]'}`}>{val > 0 ? '+' : ''}{val.toFixed(2)}x</span>
              </div>
              <input
                type="range"
                className="liquid-slider"
                min={driver.min}
                max={driver.max}
                step={driver.step}
                value={val}
                style={{ '--fill': `${percent}%` } as any}
                onChange={e => setWeatherComponentSlider(driver.key as keyof WeatherComponentSliders, Number(e.target.value), selectedBlocks)}
              />
            </div>
          );
        })}
        <button
          className={`disclosure-trigger ${showAdvanced ? 'active' : ''}`}
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{ marginTop: '2px' }}
        >
          <Sliders size={11} />
          <span className="text-[10px]">{showAdvanced ? 'Hide' : 'Show'} Advanced</span>
          {showAdvanced ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
        </button>
      </div>
    );
  }

  if (part === "sliders") {
    const SCENARIO_PRESETS = [
      { name: "Heatwave (+5°C)", icon: "🔥", temp: 2.5, humidity: -0.5, precipitation: 0, wind: -0.3, desc: "Extreme heat scenario" },
      { name: "Monsoon Onset", icon: "🌧️", temp: -1.0, humidity: 2.0, precipitation: 3.0, wind: 1.0, desc: "Heavy rain + cooling" },
      { name: "Cold Wave (-5°C)", icon: "❄️", temp: -2.5, humidity: 0.5, precipitation: 0, wind: 0.5, desc: "Winter heating demand" },
      { name: "Clear & Calm", icon: "☀️", temp: 0, humidity: 0, precipitation: 0, wind: 0, desc: "Reset to baseline" },
      { name: "Industrial Shutdown", icon: "🏭", temp: 0, humidity: 0, precipitation: 0, wind: 0, desc: "30% industrial reduction" },
    ];

    const applyScenario = (preset: typeof SCENARIO_PRESETS[0]) => {
      setWeatherComponentSlider("temperature", preset.temp, selectedBlocks);
      setWeatherComponentSlider("humidity", preset.humidity, selectedBlocks);
      setWeatherComponentSlider("precipitation", preset.precipitation, selectedBlocks);
      setWeatherComponentSlider("wind", preset.wind, selectedBlocks);
    };

    return (
      <div className="glass-panel p-4 flex flex-col gap-5">
        {/* Scenario Presets */}
        <div className="flex flex-col gap-2">
          <span className="text-[10px] uppercase tracking-widest text-[var(--muted)]">Quick Scenarios</span>
          <div className="grid grid-cols-2 gap-2">
            {SCENARIO_PRESETS.map((preset) => (
              <button
                key={preset.name}
                className="p-2 rounded-lg bg-white/5 border border-[var(--outline)] hover:bg-white/10 hover:border-[var(--accent)] transition-all text-left"
                onClick={() => applyScenario(preset)}
                title={preset.desc}
              >
                <div className="text-xs font-medium">{preset.icon} {preset.name}</div>
                <div className="text-[9px] text-[var(--muted)]">{preset.desc}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="border-t border-[var(--outline)] pt-3">
          <span className="text-[10px] uppercase tracking-widest text-[var(--muted)]">Manual Adjustment</span>
        </div>

        {WEATHER_COMPONENT_CONFIG.map((driver) => {
          const val = Number(weatherComponentSliders[driver.key as keyof WeatherComponentSliders] || 0);
          const percent = getPercentage(val, driver.min, driver.max);
          return (
            <div key={driver.key} className="sim-control-item">
              <div className="sim-control-head">
                <label className="capitalize">{driver.label}</label>
                <strong>{val > 0 ? '+' : ''}{val.toFixed(2)}x</strong>
              </div>
              <input
                type="range"
                className="liquid-slider"
                min={driver.min}
                max={driver.max}
                step={driver.step}
                value={val}
                style={{ '--fill': `${percent}%` } as any}
                onChange={e => setWeatherComponentSlider(driver.key as keyof WeatherComponentSliders, Number(e.target.value), selectedBlocks)}
              />
            </div>
          );
        })}

        <button
          className={`disclosure-trigger mt-2 ${showAdvanced ? 'active' : ''}`}
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          <Sliders size={12} />
          {showAdvanced ? 'Hide' : 'Show'} Advanced Drivers
          {showAdvanced ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </button>

        {showAdvanced && (
          <div className="disclosure-content">
            {ADVANCED_DRIVERS.map((driver) => {
              const val = Number(weatherComponentSliders[driver.key as keyof WeatherComponentSliders] || 0);
              const percent = getPercentage(val, driver.min, driver.max);
              return (
                <div key={driver.key} className="sim-control-item">
                  <div className="sim-control-head">
                    <label className="text-[11px] text-[var(--muted)]">{driver.label}</label>
                    <strong className="text-xs">{val.toFixed(2)}x</strong>
                  </div>
                  <input
                    type="range"
                    className="liquid-slider"
                    min={driver.min}
                    max={driver.max}
                    step={driver.step}
                    value={val}
                    style={{ '--fill': `${percent}%` } as any}
                    onChange={e => setWeatherComponentSlider(driver.key as keyof WeatherComponentSliders, Number(e.target.value), selectedBlocks)}
                  />
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  }

  if (part === "command") {
    return (
      <div className="flex justify-between items-center gap-4">
        <div className="flex gap-4">
          <div className="flex flex-col">
            <span className="text-[10px] uppercase text-[var(--muted)]">Status</span>
            <span className={`text-sm font-bold ${pendingRecompute ? 'text-[var(--warning)]' : 'text-[var(--success)]'}`}>
              {pendingRecompute ? 'Pending Recalc' : 'In Sync'}
            </span>
          </div>
        </div>
        <button
          className={`primary-btn px-8 flex items-center gap-2 transform transition-all active:scale-95 ${commandBusy || !pendingRecompute ? 'opacity-50 cursor-not-allowed' : 'hover:shadow-[0_0_20px_rgba(0,240,255,0.3)]'}`}
          disabled={commandBusy || !pendingRecompute}
          onClick={() => void handleRunCommand()}
        >
          {commandBusy ? <Activity className="animate-spin" size={16} /> : <Zap size={16} />}
          Commit Strategy
        </button>
      </div>
    );
  }

  if (part === "insights") {
    return (
      <div className="flex flex-col gap-3">
        <div className="glass-panel p-4 flex flex-col gap-3">
          <div className="flex justify-between items-center border-b border-[var(--outline)] pb-2">
            <span className="text-xs font-bold text-[var(--accent)]">Confidence</span>
            <span className="text-sm font-mono">{(exogInsight.weightConfidence * 100).toFixed(1)}%</span>
          </div>
          <div className="flex flex-col gap-2">
            <span className="text-[10px] uppercase text-[var(--muted)]">Impact Summary</span>
            <div className="bg-white/5 p-3 rounded-lg border border-[var(--outline)]">
              <p className="text-xs leading-relaxed text-[var(--muted)]">
                Atmospheric pressure across the {selectedBlocks.length} selected blocks indicates a
                <span className="text-white font-medium"> {exogInsight.weatherTotal >= 0 ? 'bullish' : 'bearish'} </span>
                trend of <span className="text-[var(--accent)] font-bold">{Math.abs(exogInsight.weatherTotal).toFixed(2)}%</span>.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (part === "attribution") {
    const items = [
      { label: 'Atmospheric', value: attributionSummary.weather, color: 'var(--accent)', icon: '🌤️' },
      { label: 'Temporal', value: attributionSummary.calendar, color: 'var(--warning)', icon: '📅' },
      { label: 'Strategic', value: attributionSummary.manual, color: 'var(--success)', icon: '🎯' }
    ];
    const maxAbs = Math.max(...items.map(i => Math.abs(i.value)), 1);
    return (
      <div className="glass-panel p-3 flex flex-col gap-2">
        {items.map(item => {
          const barPct = Math.min(Math.abs(item.value) / maxAbs * 100, 100);
          return (
            <div key={item.label} className="rounded-lg border border-[var(--outline)] p-2.5 bg-white/[0.02]">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[11px] text-[var(--muted)]">{item.icon} {item.label}</span>
                <span className="text-xs font-bold font-mono" style={{ color: item.color }}>{item.value >= 0 ? '+' : ''}{item.value.toFixed(1)} MW</span>
              </div>
              <div className="h-1 rounded-full bg-white/5">
                <div className="h-full rounded-full transition-all" style={{ width: `${Math.max(barPct, 2)}%`, background: item.color }} />
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  if (part === "patterns") {
    const defaultBands = [
      { key: 'peak', title: 'Peak Hours', label: 'B30-B45', time: '07:15 – 11:00', color: 'var(--warning)', icon: '⚡' },
      { key: 'low', title: 'Valley Hours', label: 'B1-B12', time: '00:00 – 03:00', color: 'var(--accent)', icon: '📉' },
      { key: 'ramp', title: 'Ramp Up', label: 'B18-B24', time: '04:15 – 06:00', color: 'var(--success)', icon: '📈' }
    ];
    return (
      <div className="glass-panel p-4 flex flex-col gap-3">
        {defaultBands.map(card => (
          <div
            key={card.key}
            className="flex items-center gap-3 p-3 rounded-lg border border-[var(--outline)] bg-white/5"
            style={{ borderLeftWidth: '3px', borderLeftColor: card.color }}
          >
            <span className="text-lg">{card.icon}</span>
            <div className="flex-1">
              <div className="text-xs font-semibold" style={{ color: card.color }}>{card.title}</div>
              <div className="text-[10px] text-[var(--muted)]">{card.time}</div>
            </div>
            <div className="text-right">
              <strong className="text-sm font-mono">{card.label}</strong>
            </div>
          </div>
        ))}
      </div>
    );
  }

  return null;
};
