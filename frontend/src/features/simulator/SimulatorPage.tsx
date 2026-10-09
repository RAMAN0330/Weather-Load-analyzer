import React, { useEffect, useMemo, useState } from "react";
import { ForecastChart } from "./components/ForecastChart";
import { DriverEditorPanel } from "./components/DriverEditorPanel";
import { ScenarioCompareModal } from "./components/ScenarioCompareModal";
import { BlockGridEditorModal } from "./components/BlockGridEditorModal";
import { useSimulatorStore } from "./store";
import { loadSimulatorDateOptionsApi } from "./simulatorDataService";
import HorizonToggle from "../../components/HorizonToggle";
import {
  PageShell as VpPageShell,
  PageHeader as VpPageHeader,
  Pill as VpPill,
} from "../../components/page/PagePrimitives.jsx";


type Props = {
  requestedDate?: string;
  baselineDays?: number;
  horizon?: 't1' | 't2';
  setHorizon?: (h: 't1' | 't2') => void;
  t2Date?: string;
};

const SIMULATOR_TABS = [
  { id: "simulation", label: "Simulation" },
  { id: "impact", label: "Impact" },
] as const;

type SimulatorTabId = typeof SIMULATOR_TABS[number]["id"];

const S = {
  page: {
    fontFamily: "'IBM Plex Mono', monospace",
    color: "#ECEEF3",
    minHeight: 0,
    height: "100%",
    overflowX: "hidden" as const,
    overflowY: "auto" as const,
    display: "flex",
    flexDirection: "column" as const,
    gap: 14,
    padding: "8px 0 16px",
  },
  demoBanner: {
    margin: "0 16px 12px",
    padding: "10px 14px",
    borderRadius: 8,
    border: "1px solid #FBBF24",
    background: "rgba(251, 191, 36, 0.12)",
    color: "#FBBF24",
    fontSize: 12,
  },
  kpiGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
    gap: 12,
    padding: "0 16px",
  },
  kpiCard: {
    minHeight: 132,
    padding: "16px 18px",
    background: "linear-gradient(180deg, rgba(30, 29, 35, 0.98), rgba(22, 22, 27, 0.98))",
    borderRadius: 14,
    border: "1px solid #2A292F",
    display: "flex",
    flexDirection: "column" as const,
    justifyContent: "space-between" as const,
    gap: 8,
    boxShadow: "0 18px 40px rgba(0, 0, 0, 0.18)",
  },
  kpiLabel: {
    fontSize: 9,
    textTransform: "uppercase" as const,
    letterSpacing: 1.5,
    color: "#6B7186",
  },
  kpiValue: (color = "#ECEEF3") => ({
    fontSize: 30,
    lineHeight: 1.05,
    fontWeight: 700,
    color,
  }),
  kpiUnit: {
    fontSize: 11,
    fontWeight: 400,
    opacity: 0.5,
  },
  kpiSub: {
    fontSize: 10,
    color: "#6B7186",
    lineHeight: 1.45,
  },
  card: {
    background: "linear-gradient(180deg, rgba(26, 25, 30, 0.98), rgba(20, 20, 24, 0.96))",
    borderRadius: 16,
    border: "1px solid #2A292F",
    overflow: "hidden" as const,
    display: "flex",
    flexDirection: "column" as const,
    boxShadow: "0 18px 40px rgba(0, 0, 0, 0.18)",
  },
  cardTitle: {
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 1.2,
    color: "#A0A5B8",
    textTransform: "uppercase" as const,
  },
  workspace: {
    ...{
      background: "linear-gradient(180deg, rgba(26, 25, 30, 0.98), rgba(20, 20, 24, 0.96))",
      borderRadius: 16,
      border: "1px solid #2A292F",
      overflow: "visible" as const,
      display: "flex",
      flexDirection: "column" as const,
      boxShadow: "0 18px 40px rgba(0, 0, 0, 0.18)",
    },
    margin: "0 16px",
  },
  tabBar: {
    display: "inline-flex",
    gap: 4,
    padding: 5,
    background: "#141419",
    border: "1px solid #2A292F",
    borderRadius: 999,
    flexWrap: "wrap" as const,
  },
  tab: (active: boolean) => ({
    padding: "8px 16px",
    fontSize: 10,
    fontWeight: 600,
    borderRadius: 999,
    border: "1px solid transparent",
    background: active ? "rgba(240, 120, 37, 0.14)" : "transparent",
    color: active ? "#F07825" : "#A0A5B8",
    cursor: "pointer",
    fontFamily: "inherit",
  }),
  actionBtn: {
    fontSize: 10,
    fontWeight: 700,
    padding: "8px 14px",
    borderRadius: 999,
    border: "1px solid #2A292F",
    background: "#1A191E",
    color: "#ECEEF3",
    cursor: "pointer",
    fontFamily: "inherit",
  },
  miniBadge: (color: string) => ({
    fontSize: 9,
    fontWeight: 700,
    padding: "4px 10px",
    borderRadius: 999,
    background: `${color}18`,
    color,
    border: `1px solid ${color}33`,
  }),
};

const KpiCard = ({
  label,
  value,
  unit,
  sub,
  tone,
}: {
  label: string;
  value: React.ReactNode;
  unit?: string;
  sub: string;
  tone?: string;
}) => (
  <div style={S.kpiCard}>
    <div style={S.kpiLabel}>{label}</div>
    <div>
      <span style={S.kpiValue(tone)}>{value}</span>
      {unit ? <span style={S.kpiUnit}> {unit}</span> : null}
    </div>
    <div style={S.kpiSub}>{sub}</div>
  </div>
);

export const SimulatorPage: React.FC<Props> = ({
  requestedDate,
  baselineDays = 7,
  horizon = 't1',
  setHorizon,
  t2Date,
}) => {
  const comparePayload = useSimulatorStore((s) => s.comparePayload);
  const closeCompare = useSimulatorStore((s) => s.closeCompare);
  const loadFromFileData = useSimulatorStore((s) => s.loadFromFileData);
  const blocks = useSimulatorStore((s) => s.blocks);
  const selectedBlocks = useSimulatorStore((s) => s.selectedBlocks);
  const dataDate = useSimulatorStore((s) => s.dataDate);
  const dataStatus = useSimulatorStore((s) => s.dataStatus);
  const dataError = useSimulatorStore((s) => s.dataError);
  const weatherDeltasByBlock = useSimulatorStore((s) => s.weatherDeltasByBlock);
  const [gridOpen, setGridOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<SimulatorTabId>("simulation");
  const [overlayPanel, setOverlayPanel] = useState<string | null>(null);
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

  const exogInsight = useMemo(() => {
    if (!selected.length) return { weatherTotal: 0, weightConfidence: 0 };
    const n = Math.max(1, selected.length);
    const weatherTotal = selected.reduce(
      (sum, block) => sum + Number(block.exog?.weather_total_pct ?? block.drivers.weather_pct ?? 0),
      0
    ) / n;
    const weightConfidence = selected.reduce(
      (sum, block) => sum + Number(block.exog?.weight_confidence || 0),
      0
    ) / n;
    const round3 = (value: number) => Math.round((Number(value) || 0) * 1000) / 1000;
    return {
      weatherTotal: round3(weatherTotal),
      weightConfidence: round3(weightConfidence),
    };
  }, [selected]);

  const selectionCount = selectedBlocks.length || blocks.length;
  const peakDeltaMw = Math.abs(netSummary.peakFinal.value - netSummary.peakBaseline.value);
  const peakShiftBlocks = netSummary.peakFinal.block && netSummary.peakBaseline.block
    ? netSummary.peakFinal.block - netSummary.peakBaseline.block
    : 0;

  // Reserve Margin Impact: remaining reserve % after scenario peak vs installed capacity.
  // Capacity comes from block metadata when available; otherwise shown as N/A.
  const installedCapacityMw: number | null =
    (blocks[0] as any)?.region_capacity_mw ?? null;
  const scenarioPeak = netSummary.peakFinal.value || 0;
  const reserveMarginPct = installedCapacityMw && scenarioPeak > 0
    ? Math.round(((installedCapacityMw - scenarioPeak) / installedCapacityMw) * 100)
    : null;
  const reserveBreach = reserveMarginPct != null && reserveMarginPct < 15;

  // Ramp Feasibility: count blocks where scenario creates ramp > 150 MW/15min
  const RAMP_LIMIT_MW = 150;
  const rampFeasibilityViolations = useMemo(() => {
    if (scopeBlocks.length < 2) return 0;
    let count = 0;
    for (let i = 1; i < scopeBlocks.length; i++) {
      const ramp = Math.abs((scopeBlocks[i].final_mw || 0) - (scopeBlocks[i - 1].final_mw || 0));
      if (ramp > RAMP_LIMIT_MW) count++;
    }
    return count;
  }, [scopeBlocks]);
  const confidencePct = exogInsight.weightConfidence * 100;
  const weatherTone = exogInsight.weatherTotal >= 0 ? "#F07825" : "#5B9FE4";
  const weatherSignalLabel = exogInsight.weatherTotal >= 0 ? "Weather Uplift" : "Weather Drag";
  const simulatorActionButtons = activeTab === "simulation"
    ? [
        { id: "adjustments", label: "Open Weather Adjustments" },
        { id: "scenarios", label: "Open Quick Scenarios" },
        { id: "patterns", label: "Open Pattern Windows" },
      ]
    : [
        { id: "insights", label: "Open Analytic Insights" },
        { id: "attribution", label: "Open Variance Attribution" },
        { id: "adjustments", label: "Open Weather Adjustments" },
      ];

  return (
    <>
      <VpPageShell className="simulator-page">
        {setHorizon && (
          <div className="vp-horizon-floater">
            <HorizonToggle horizon={horizon} setHorizon={setHorizon} t2Date={t2Date} />
          </div>
        )}
        {dataStatus === "demo" && (
          <div role="alert" style={S.demoBanner}>
            DEMO DATA: the blocks below are synthetic, not real load.
            {dataError ? ` ${dataError}` : ""}
          </div>
        )}
        <div style={S.kpiGrid}>
          <KpiCard
            label="Net Shift"
            value={`${netSummary.shift >= 0 ? "+" : ""}${netSummary.shift.toFixed(1)}`}
            unit="MW"
            sub={`${selectionCount} blocks in scope • baseline ${netSummary.baseline.toFixed(1)} MW`}
            tone={netSummary.shift >= 0 ? "#34D399" : "#F87171"}
          />
          <KpiCard
            label="Energy Delta"
            value={netSummary.energy.toFixed(1)}
            unit="MWh"
            sub={`Final energy impact from current edits across selected scope`}
            tone={Math.abs(netSummary.energy) <= 25 ? "#34D399" : "#FBBF24"}
          />
          <KpiCard
            label="Peak Delta"
            value={peakDeltaMw.toFixed(1)}
            unit="MW"
            sub={`Peak moved ${peakShiftBlocks > 0 ? "+" : ""}${peakShiftBlocks} blocks from baseline`}
            tone={peakDeltaMw <= 40 ? "#34D399" : peakDeltaMw <= 90 ? "#FBBF24" : "#F87171"}
          />
          <KpiCard
            label={weatherSignalLabel}
            value={`${exogInsight.weatherTotal >= 0 ? "+" : ""}${Math.abs(exogInsight.weatherTotal).toFixed(2)}`}
            unit="%"
            sub={`Average exogenous weather pressure across selected blocks`}
            tone={weatherTone}
          />
          <KpiCard
            label="Weight Confidence"
            value={confidencePct.toFixed(1)}
            unit="%"
            sub={`${seasonLabel} regime • ${calendarContextLabel}`}
            tone={confidencePct >= 70 ? "#34D399" : confidencePct >= 45 ? "#FBBF24" : "#F87171"}
          />
          <KpiCard
            label="Scenario Scope"
            value={selectionCount}
            unit="blocks"
            sub={selectionLabel}
            tone="#ECEEF3"
          />
          <KpiCard
            label="Reserve Margin"
            value={reserveMarginPct != null ? reserveMarginPct : "--"}
            unit="%"
            sub={reserveBreach ? `⚠ Breach — peak ${Math.round(scenarioPeak)} MW exceeds safe limit` : scenarioPeak > 0 ? `Scenario peak ${Math.round(scenarioPeak)} MW` : "No scenario peak"}
            tone={reserveMarginPct == null ? "#ECEEF3" : reserveBreach ? "#F87171" : reserveMarginPct < 20 ? "#FBBF24" : "#34D399"}
          />
          <KpiCard
            label="Ramp Feasibility"
            value={rampFeasibilityViolations}
            unit="violations"
            sub={rampFeasibilityViolations === 0 ? `All ramps within ${RAMP_LIMIT_MW} MW/15min` : `${rampFeasibilityViolations} blocks exceed ${RAMP_LIMIT_MW} MW/15min`}
            tone={rampFeasibilityViolations === 0 ? "#34D399" : rampFeasibilityViolations <= 3 ? "#FBBF24" : "#F87171"}
          />
        </div>

        <div style={S.workspace}>
          <div style={{ padding: "18px 20px 14px", borderBottom: "1px solid #2A292F" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
              <div style={S.tabBar}>
                {SIMULATOR_TABS.map((tab) => (
                  <button key={tab.id} type="button" onClick={() => setActiveTab(tab.id)} style={S.tab(activeTab === tab.id)}>
                    {tab.label}
                  </button>
                ))}
              </div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "flex-end", alignItems: "center" }}>
                <span style={S.miniBadge("#34D399")}>{seasonLabel}</span>
                <span style={S.miniBadge("#5B9FE4")}>{calendarContextLabel}</span>
                <span style={{ ...S.miniBadge("#F07825"), maxWidth: 280, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{selectionLabel}</span>
                {[
                  { label: "Choose Date", sub: "Pick forecast date", color: "#FBBF24", onClick: () => setCalendarPanelOpen(true) },
                  { label: "Block Grid", sub: "96-block schedule", color: "#5B9FE4", onClick: () => setGridOpen(true) },
                  ...(activeTab === "simulation"
                    ? [
                        { label: "Weather Adj.", sub: "Temp · Humidity · Cloud", color: "#34D399", onClick: () => setOverlayPanel("adjustments") },
                        { label: "Quick Scenarios", sub: "Preset load scenarios", color: "#C084FC", onClick: () => setOverlayPanel("scenarios") },
                        { label: "Pattern Windows", sub: "Shape & period filters", color: "#F07825", onClick: () => setOverlayPanel("patterns") },
                      ]
                    : [
                        { label: "Analytic Insights", sub: "Model diagnostics", color: "#34D399", onClick: () => setOverlayPanel("insights") },
                        { label: "Variance Attribution", sub: "Error breakdown", color: "#C084FC", onClick: () => setOverlayPanel("attribution") },
                        { label: "Weather Adj.", sub: "Temp · Humidity · Cloud", color: "#5B9FE4", onClick: () => setOverlayPanel("adjustments") },
                      ]
                  ),
                ].map(({ label, sub, color, onClick }) => (
                  <button
                    key={label}
                    type="button"
                    onClick={onClick}
                    style={{
                      display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 5,
                      minWidth: 130, padding: "10px 14px", borderRadius: 14,
                      border: `1px solid ${color}33`,
                      background: `${color}10`,
                      color: "#ECEEF3",
                      cursor: "pointer", fontFamily: "inherit", textAlign: "left", transition: "all 0.15s ease",
                    }}
                  >
                    <span style={{ fontSize: 8, letterSpacing: 1.1, textTransform: "uppercase", color: "#6B7186" }}>Quick panel</span>
                    <span style={{ fontSize: 12, fontWeight: 700, color }}>{label}</span>
                    <span style={{ fontSize: 9, color: "#6B7186" }}>{sub}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1.5fr) minmax(300px, 0.78fr)", gap: 12, padding: "14px 16px", alignItems: "start" }}>
            <div style={{ ...S.card, minWidth: 0, boxShadow: "none" }}>
              <div style={{ padding: "14px 16px", borderBottom: "1px solid #2A292F", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                <div>
                  <div style={S.cardTitle}>Forecast Canvas</div>
                </div>
                <div style={{ fontSize: 10, color: "#A0A5B8" }}>
                  Peak {netSummary.peakBaseline.block || "--"} {"->"} {netSummary.peakFinal.block || "--"} • {peakDeltaMw.toFixed(1)} MW delta
                </div>
              </div>
              <div
                style={{
                  padding: 10,
                  height: "clamp(430px, 52vh, 540px)",
                  minHeight: 430,
                  overflow: "hidden",
                  display: "flex",
                  flexDirection: "column",
                  width: "100%",
                  minWidth: 0,
                }}
              >
                <ForecastChart />
              </div>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <DriverEditorPanel activeTab={activeTab} part="command" />

              <div>
                <div style={{ ...S.cardTitle, padding: "0 4px 8px" }}>Target Scope</div>
                <DriverEditorPanel
                  activeTab={activeTab}
                  part="selection"
                  calendarPanelOpen={calendarPanelOpen}
                  onCalendarPanelOpenChange={setCalendarPanelOpen}
                  selectedDate={selectedDate}
                  calendarContextLabel={calendarContextLabel}
                />
              </div>

              <div style={S.card}>
                <div style={{ padding: "14px 16px", borderBottom: "1px solid #2A292F" }}>
                  <div style={S.cardTitle}>Scenario Snapshot</div>
                </div>
                <div style={{ padding: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                  <div style={{ background: "#201F25", border: "1px solid #2A292F", borderRadius: 12, padding: "12px 14px" }}>
                    <div style={{ fontSize: 8, letterSpacing: 1.2, textTransform: "uppercase", color: "#6B7186", marginBottom: 6 }}>Temperature</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: deltaSummary.temp >= 0 ? "#F07825" : "#5B9FE4" }}>
                      {deltaSummary.temp >= 0 ? "+" : ""}{deltaSummary.temp.toFixed(2)} C
                    </div>
                  </div>
                  <div style={{ background: "#201F25", border: "1px solid #2A292F", borderRadius: 12, padding: "12px 14px" }}>
                    <div style={{ fontSize: 8, letterSpacing: 1.2, textTransform: "uppercase", color: "#6B7186", marginBottom: 6 }}>Humidity</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: "#45b7d1" }}>
                      {deltaSummary.hum >= 0 ? "+" : ""}{deltaSummary.hum.toFixed(2)} %
                    </div>
                  </div>
                  <div style={{ background: "#201F25", border: "1px solid #2A292F", borderRadius: 12, padding: "12px 14px" }}>
                    <div style={{ fontSize: 8, letterSpacing: 1.2, textTransform: "uppercase", color: "#6B7186", marginBottom: 6 }}>Rain</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: "#C084FC" }}>
                      {deltaSummary.rain >= 0 ? "+" : ""}{deltaSummary.rain.toFixed(2)} mm
                    </div>
                  </div>
                  <div style={{ background: "#201F25", border: "1px solid #2A292F", borderRadius: 12, padding: "12px 14px" }}>
                    <div style={{ fontSize: 8, letterSpacing: 1.2, textTransform: "uppercase", color: "#6B7186", marginBottom: 6 }}>Wind</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: "#34D399" }}>
                      {deltaSummary.wind >= 0 ? "+" : ""}{deltaSummary.wind.toFixed(2)} m/s
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </VpPageShell>
      <ScenarioCompareModal open={Boolean(comparePayload)} payload={comparePayload} onClose={closeCompare} />
      <BlockGridEditorModal open={gridOpen} onClose={() => setGridOpen(false)} selectionLabel={selectionLabel} />

      {overlayPanel && (
        <div className="modal-overlay" onClick={() => setOverlayPanel(null)}>
          <div className="modal-content" style={{ width: "min(920px, 100%)", maxHeight: "84vh", borderRadius: 16 }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header" style={{ padding: "16px 18px", borderBottom: "1px solid #2A292F" }}>
              <div>
                <h3 style={{ margin: 0 }}>
                  {overlayPanel === "adjustments" && "Weather Adjustments"}
                  {overlayPanel === "scenarios" && "Quick Scenarios"}
                  {overlayPanel === "patterns" && "Pattern Windows"}
                  {overlayPanel === "insights" && "Analytic Insights"}
                  {overlayPanel === "attribution" && "Variance Attribution"}
                </h3>
              </div>
              <button type="button" onClick={() => setOverlayPanel(null)} style={{ background: "none", border: "none", color: "#6B7186", fontSize: 20, cursor: "pointer" }}>×</button>
            </div>
            <div className="modal-body">
              {overlayPanel === "adjustments" && <DriverEditorPanel activeTab={activeTab} part="adjustments" />}
              {overlayPanel === "scenarios" && <DriverEditorPanel activeTab={activeTab} part="scenarios" />}
              {overlayPanel === "patterns" && <DriverEditorPanel activeTab={activeTab} part="patterns" />}
              {overlayPanel === "insights" && <DriverEditorPanel activeTab={activeTab} part="insights" />}
              {overlayPanel === "attribution" && <DriverEditorPanel activeTab={activeTab} part="attribution" />}
            </div>
          </div>
        </div>
      )}

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
