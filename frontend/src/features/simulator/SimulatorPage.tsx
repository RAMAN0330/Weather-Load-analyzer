import React, { useEffect, useMemo, useState } from "react";
import { ForecastChart } from "./components/ForecastChart";
import { DriverEditorPanel } from "./components/DriverEditorPanel";
import { ScenarioCompareModal } from "./components/ScenarioCompareModal";
import { BlockGridEditorModal } from "./components/BlockGridEditorModal";
import { useSimulatorStore } from "./store";
import { loadSimulatorDateOptionsApi } from "./simulatorDataService";
import { blockTimeLabel } from "./utils";

type Props = {
  requestedDate?: string;
  baselineDays?: number;
};

const SIMULATOR_TABS = [
  { id: "simulation", label: "Simulation" },
  { id: "impact", label: "Impact" },
] as const;

type SimulatorTabId = typeof SIMULATOR_TABS[number]["id"];

export const SimulatorPage: React.FC<Props> = ({ requestedDate, baselineDays = 7 }) => {
  const comparePayload = useSimulatorStore((s) => s.comparePayload);
  const closeCompare = useSimulatorStore((s) => s.closeCompare);
  const loadFromFileData = useSimulatorStore((s) => s.loadFromFileData);
  const blocks = useSimulatorStore((s) => s.blocks);
  const selectedBlocks = useSimulatorStore((s) => s.selectedBlocks);
  const weightsSource = useSimulatorStore((s) => s.weightsSource);
  const partialDayBiasCorrection = useSimulatorStore((s) => s.partialDayBiasCorrection);
  const dataDate = useSimulatorStore((s) => s.dataDate);
  const weatherDeltasByBlock = useSimulatorStore((s) => s.weatherDeltasByBlock);
  const [gridOpen, setGridOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<SimulatorTabId>("simulation");
  const [calendarPanelOpen, setCalendarPanelOpen] = useState(false);
  const [calendarDraftDate, setCalendarDraftDate] = useState("");
  const [dateOptions, setDateOptions] = useState<string[]>([]);
  const [selectedDate, setSelectedDate] = useState("");

  const parseYmd = (raw: string) => {
    const text = String(raw || "").trim();
    if (!text) return null;
    const parts = text.split("-").map((x) => Number(x));
    if (parts.length !== 3 || parts.some((x) => !Number.isFinite(x))) return null;
    return new Date(parts[0], parts[1] - 1, parts[2]);
  };

  const toYmd = (dt: Date) => {
    const y = dt.getFullYear();
    const m = String(dt.getMonth() + 1).padStart(2, "0");
    const d = String(dt.getDate()).padStart(2, "0");
    return `${y}-${m}-${d}`;
  };

  const [calendarMonthCursor, setCalendarMonthCursor] = useState<Date>(() => {
    const base = parseYmd(selectedDate || dataDate || "") || new Date();
    return new Date(base.getFullYear(), base.getMonth(), 1);
  });

  const availableDateSet = useMemo(() => new Set(dateOptions), [dateOptions]);

  const calendarMonthLabel = useMemo(
    () => calendarMonthCursor.toLocaleDateString(undefined, { month: "long", year: "numeric" }),
    [calendarMonthCursor]
  );

  const calendarCells = useMemo(() => {
    const monthStart = new Date(calendarMonthCursor.getFullYear(), calendarMonthCursor.getMonth(), 1);
    const firstCellDate = new Date(monthStart);
    firstCellDate.setDate(1 - monthStart.getDay());
    const todayYmd = toYmd(new Date());
    const selectedYmd = calendarDraftDate || selectedDate || dataDate || "";
    return Array.from({ length: 42 }, (_, idx) => {
      const dt = new Date(firstCellDate);
      dt.setDate(firstCellDate.getDate() + idx);
      const ymd = toYmd(dt);
      const inMonth = dt.getMonth() === calendarMonthCursor.getMonth();
      const hasData = dateOptions.length === 0 || availableDateSet.has(ymd);
      return {
        dt,
        ymd,
        inMonth,
        available: true,
        hasData,
        isToday: ymd === todayYmd,
        isSelected: ymd === selectedYmd,
      };
    });
  }, [calendarMonthCursor, dateOptions.length, availableDateSet, calendarDraftDate, selectedDate, dataDate]);

  const calendarWeekDays = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"];

  const selectionLabel = useMemo(() => {
    if (!selectedBlocks.length) return "No selection";
    if (selectedBlocks.length >= 96) return "Selected: All Blocks (96)";
    if (selectedBlocks.length === 1) return `Selected: Block ${selectedBlocks[0]}`;
    return `Selected: Blocks ${selectedBlocks[0]}-${selectedBlocks[selectedBlocks.length - 1]} (${selectedBlocks.length})`;
  }, [selectedBlocks]);

  const selected = useMemo(() => blocks.filter((b) => selectedBlocks.includes(b.block_number)), [blocks, selectedBlocks]);
  const scopeBlocks = useMemo(() => (selected.length ? selected : blocks), [selected, blocks]);

  const netSummary = useMemo(() => {
    if (!scopeBlocks.length) {
      return {
        baseline: 0,
        final: 0,
        shift: 0,
        energy: 0,
        peakBaseline: { block: 0, value: 0 },
        peakFinal: { block: 0, value: 0 },
      };
    }
    let baseline = 0;
    let final = 0;
    let peakBaseline = { block: 0, value: -Infinity };
    let peakFinal = { block: 0, value: -Infinity };

    scopeBlocks.forEach((b) => {
      const bValue = Number(b.baseline_mw || 0);
      const fValue = Number(b.final_mw || 0);
      baseline += bValue;
      final += fValue;
      if (bValue > peakBaseline.value) {
        peakBaseline = { block: b.block_number, value: bValue };
      }
      if (fValue > peakFinal.value) {
        peakFinal = { block: b.block_number, value: fValue };
      }
    });

    return {
      baseline,
      final,
      shift: final - baseline,
      energy: (final - baseline) * 0.25,
      peakBaseline,
      peakFinal,
    };
  }, [scopeBlocks]);

  useEffect(() => {
    loadFromFileData(requestedDate, baselineDays);
    // Intentionally ignore baselineDays live changes to avoid repeated auto-refresh/recompute loops.
    // Baseline should refresh via explicit date change or manual command actions.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadFromFileData, requestedDate]);

  useEffect(() => {
    if (activeTab !== "simulation") {
      setCalendarPanelOpen(false);
    }
  }, [activeTab]);

  useEffect(() => {
    const run = async () => {
      try {
        const cfg = await loadSimulatorDateOptionsApi();
        const opts = cfg?.all_dates || cfg?.dates || [];
        setDateOptions(opts);
        const pref = requestedDate || dataDate || opts[opts.length - 1] || "";
        setSelectedDate(pref);
      } catch {
        setDateOptions([]);
      }
    };
    void run();
  }, [dataDate, requestedDate]);

  const handleDateChange = async (next: string) => {
    if (!next) return;
    setSelectedDate(next);
    await loadFromFileData(next, baselineDays || 7);
  };

  const submitCalendarSelection = async () => {
    await handleDateChange(calendarDraftDate);
    setCalendarPanelOpen(false);
  };

  const calendarContextLabel = useMemo(() => {
    const raw = String(selectedDate || dataDate || "").trim();
    if (!raw) return "--";
    const dt = new Date(raw);
    if (Number.isNaN(dt.getTime())) return "--";
    const dayName = dt.toLocaleDateString(undefined, { weekday: "long" });
    const dayType = (dt.getDay() === 0 || dt.getDay() === 6) ? "Weekend" : "Weekday";
    return `${dayName} (${dayType})`;
  }, [selectedDate, dataDate]);

  const seasonLabel = useMemo(() => {
    const raw = String(dataDate || "").trim();
    if (!raw) return "unknown";
    const dt = new Date(raw);
    if (Number.isNaN(dt.getTime())) return "unknown";
    const m = dt.getMonth() + 1;
    if (m >= 6 && m <= 9) return "Monsoon";
    if (m >= 11 || m <= 2) return "Winter";
    return "Summer";
  }, [dataDate]);

  const deltaSummary = useMemo(() => {
    const bSet = selectedBlocks.length > 0 ? selectedBlocks : scopeBlocks.map(b => b.block_number);
    if (!bSet.length) {
      return { temp: 0, hum: 0, rain: 0, wind: 0 };
    }
    let temp = 0;
    let hum = 0;
    let rain = 0;
    let wind = 0;
    bSet.forEach((b) => {
      const d = weatherDeltasByBlock[b];
      temp += Number(d?.temp_delta_c || 0);
      hum += Number(d?.humidity_delta_pct || 0);
      rain += Number(d?.precip_delta_mm || 0);
      wind += Number(d?.wind_delta_mps || 0);
    });
    const n = Math.max(1, bSet.length);
    const round2 = (v: number) => Math.round(v * 100) / 100;
    return {
      temp: round2(temp / n),
      hum: round2(hum / n),
      rain: round2(rain / n),
      wind: round2(wind / n),
    };
  }, [selectedBlocks, scopeBlocks, weatherDeltasByBlock]);
  return (
    <>
      <main className="page page-full">
        <div className="sim-shell">
          {/* Central Canvas: Visualization & Action (full width, no left sidebar) */}
          <div className="sim-main-canvas" style={{ flex: 1 }}>
            <header className="sim-top-kpi-bar glass-panel-light">
              <div className="sim-kpi-item">
                <span>Net Shift</span>
                <strong className={netSummary.shift >= 0 ? "positive" : "negative"}>
                  {netSummary.shift >= 0 ? "+" : ""}{netSummary.shift.toFixed(1)} MW
                </strong>
              </div>
              <div className="sim-kpi-item">
                <span>Energy Δ</span>
                <strong>{netSummary.energy.toFixed(1)} MWh</strong>
              </div>
              <div className="sim-kpi-item">
                <span>Peak Dev.</span>
                <strong>{Math.abs(netSummary.peakFinal.value - netSummary.peakBaseline.value).toFixed(1)} MW</strong>
              </div>
              <div className="sim-kpi-item">
                <span>Season</span>
                <strong className="capitalize">{seasonLabel}</strong>
              </div>
            </header>

            <div className="sim-forecast-panel p-0 overflow-hidden">
              <div className="p-3 border-b border-[var(--outline)] flex justify-between items-center">
                <div>
                  <h3 className="text-sm font-bold tracking-tight">Scenario Simulator</h3>
                  <span className="text-[10px] text-[var(--muted)]">Baseline vs Adjusted vs Actual (96 blocks)</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-[10px] text-[var(--muted)] uppercase tracking-widest bg-white/5 px-2 py-1 rounded">Shift + Drag to Select</span>
                </div>
              </div>
              <div className="chart-body" style={{ flex: 1, minHeight: 0, padding: '12px' }}>
                <ForecastChart />
              </div>
              <div className="p-3 border-t border-[var(--outline)] flex-shrink-0">
                <DriverEditorPanel activeTab="simulation" part="command" />
              </div>
            </div>

            {/* Quick Scenarios & Detected Patterns side by side */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
              <div>
                <span className="sim-section-label">Quick Scenarios</span>
                <DriverEditorPanel activeTab="simulation" part="scenarios" />
              </div>
              <div>
                <span className="sim-section-label">Detected Pattern Windows</span>
                <DriverEditorPanel activeTab="simulation" part="patterns" />
              </div>
            </div>
          </div>

          {/* Right Sidebar: Scope, Intelligence, Adjustments & Attribution */}
          <aside className="sim-intel-panel" style={{ width: '320px', minWidth: '320px' }}>
            <span className="sim-section-label">Target Scope</span>
            <DriverEditorPanel
              activeTab="simulation"
              part="selection"
              calendarPanelOpen={calendarPanelOpen}
              onCalendarPanelOpenChange={setCalendarPanelOpen}
              selectedDate={selectedDate}
              calendarContextLabel={calendarContextLabel}
            />

            <span className="sim-section-label">Analytic Insights</span>
            <DriverEditorPanel activeTab="simulation" part="insights" />

            <span className="sim-section-label">Weather Adjustments</span>
            <DriverEditorPanel activeTab="simulation" part="adjustments" />

            <span className="sim-section-label">Variance Attribution</span>
            <DriverEditorPanel activeTab="simulation" part="attribution" />
          </aside>
        </div>
      </main>
      <ScenarioCompareModal open={Boolean(comparePayload)} payload={comparePayload} onClose={closeCompare} />
      <BlockGridEditorModal open={gridOpen} onClose={() => setGridOpen(false)} selectionLabel={selectionLabel} />

      {calendarPanelOpen && (
        <div className="sim-modal-backdrop" onClick={() => setCalendarPanelOpen(false)}>
          <div className="sim-modal sim-calendar-modal" onClick={e => e.stopPropagation()}>
            <div className="panel-header">
              <h3>Select Simulation Date</h3>
              <button type="button" className="close-btn" onClick={() => setCalendarPanelOpen(false)}>×</button>
            </div>
            <div className="sim-calendar-widget">
              <div className="sim-calendar-month-nav">
                <button type="button" onClick={() => setCalendarMonthCursor(new Date(calendarMonthCursor.getFullYear(), calendarMonthCursor.getMonth() - 1, 1))}>&lt;</button>
                <strong>{calendarMonthLabel}</strong>
                <button type="button" onClick={() => setCalendarMonthCursor(new Date(calendarMonthCursor.getFullYear(), calendarMonthCursor.getMonth() + 1, 1))}>&gt;</button>
              </div>
              <div className="sim-calendar-weekdays">
                {calendarWeekDays.map(d => <span key={d}>{d}</span>)}
              </div>
              <div className="sim-calendar-days">
                {calendarCells.map(cell => (
                  <button
                    key={cell.ymd}
                    type="button"
                    className={`sim-calendar-day ${cell.isSelected ? "active" : ""} ${cell.inMonth ? "" : "outside"}`}
                    onClick={() => setCalendarDraftDate(cell.ymd)}
                  >
                    {cell.dt.getDate()}
                  </button>
                ))}
              </div>
              <div className="sim-modal-footer">
                <div className="sim-draft-label">Draft: {calendarDraftDate || "--"}</div>
                <button type="button" className="primary-btn" onClick={() => void submitCalendarSelection()} disabled={!calendarDraftDate}>
                  Confirm Date
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
