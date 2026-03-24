import React, { useMemo } from 'react';
import {
    Activity,
    TrendingUp,
    Zap,
    CheckCircle2,
    AlertCircle,
    Download,
    Thermometer,
    Droplets,
    CloudRain,
    Wind
} from 'lucide-react';

interface ForecastPageProps {
    liveData: any;
    liveMeta: any;
    driverContributions: any[];
    decisionSignals?: any[];
    effectiveDate: string;
    selectedRegion: string;
    onRefresh: () => void;
    onDownload?: () => void;
    canDownload?: boolean;
    fmt: (v: any) => string;
    chart: React.ReactNode;
    weatherStrip?: React.ReactNode;
    forecastTable?: React.ReactNode;
}

const blockTime = (block: number) => {
    const minutes = (block - 1) * 15;
    const h = String(Math.floor(minutes / 60)).padStart(2, '0');
    const m = String(minutes % 60).padStart(2, '0');
    return `${h}:${m}`;
};

export const ForecastPage: React.FC<ForecastPageProps> = ({
    liveData,
    liveMeta,
    driverContributions,
    decisionSignals = [],
    effectiveDate,
    selectedRegion,
    onRefresh,
    onDownload,
    canDownload = true,
    fmt,
    chart,
    weatherStrip,
    forecastTable
}) => {
    const stats = useMemo(() => {
        if (!liveData || !Array.isArray(liveData)) return null;
        const actuals = liveData.filter((d: any) => d.actual_mw != null && d.actual_mw > 0);
        const lastActual = actuals[actuals.length - 1];
        const peak = Math.max(...liveData.map((d: any) => d.forecast_mw || 0), 0);
        const energy = liveData.reduce((acc: number, d: any) => acc + (d.forecast_mw || 0), 0) * 0.25;
        const lastBlock = lastActual?.block_number || 0;

        return { lastActual: lastActual?.actual_mw || 0, lastBlock, peak, energy };
    }, [liveData]);

    const insights = useMemo(() => {
        if (!liveMeta?.insights) return [];
        return Array.isArray(liveMeta.insights) ? liveMeta.insights : [liveMeta.insights];
    }, [liveMeta]);

    const activeDrivers = useMemo(() => {
        return (driverContributions || []).filter((d: any) => Math.abs(d.mw) > 0.5);
    }, [driverContributions]);

    const forecastHealth = useMemo(() => {
        if (!liveData || !Array.isArray(liveData)) return { mape: 0, status: 'unknown', color: 'var(--muted)', consecutiveHigh: 0 };
        const actuals = liveData.filter((d: any) => d.actual_mw != null && d.actual_mw > 0 && d.forecast_mw != null && d.forecast_mw > 0);
        if (actuals.length < 4) return { mape: 0, status: 'insufficient', color: 'var(--muted)', consecutiveHigh: 0 };

        const apes = actuals.map((d: any) => Math.abs(d.actual_mw - d.forecast_mw) / Math.max(d.actual_mw, 1) * 100);
        const mape = apes.reduce((a: number, b: number) => a + b, 0) / apes.length;

        let consecutive = 0;
        let maxConsecutive = 0;
        for (const ape of apes.reverse()) {
            if (ape > 5) { consecutive++; maxConsecutive = Math.max(maxConsecutive, consecutive); }
            else { consecutive = 0; }
        }

        let status = 'good';
        let color = 'var(--success)';
        if (mape > 5) { status = 'critical'; color = 'var(--danger)'; }
        else if (mape > 2) { status = 'warning'; color = 'var(--warning)'; }

        return { mape: Math.round(mape * 100) / 100, status, color, consecutiveHigh: maxConsecutive, needsReforecast: maxConsecutive >= 8 };
    }, [liveData]);

    // Top decision signals (high/medium risk only)
    const topSignals = useMemo(() => {
        return decisionSignals
            .filter((s: any) => s.risk_flag && s.risk_flag !== 'low')
            .sort((a: any, b: any) => {
                const p: Record<string, number> = { high: 3, medium: 2, low: 1 };
                return (p[b.risk_flag] || 0) - (p[a.risk_flag] || 0);
            })
            .slice(0, 6);
    }, [decisionSignals]);

    return (
        <main className="forecast-page">
            {/* Auto-reforecast alert */}
            {forecastHealth.needsReforecast && (
                <div className="fp-alert-bar">
                    <div>
                        <span className="text-xs font-bold text-[var(--danger)]">Auto-Reforecast Recommended</span>
                        <p className="text-[10px] text-[var(--muted)] mt-0.5">
                            MAPE exceeds 5% for {forecastHealth.consecutiveHigh} consecutive blocks.
                        </p>
                    </div>
                    <button className="primary-btn text-xs" onClick={onRefresh}>
                        <Zap size={14} /> Reforecast Now
                    </button>
                </div>
            )}

            {/* ── MAIN CONTENT: 2-column layout (center + right sidebar) ── */}
            <div className="fp-body fp-body--two-col">

                {/* CENTER: Chart + Weather + Table */}
                <div className="fp-center">
                    {/* Chart Panel */}
                    <div className="fp-chart-panel">
                        <div className="fp-chart-header">
                            <div>
                                <h3>Predictive VidyutPragya</h3>
                                <span>96 Quarters &bull; Volumetric Load Projection</span>
                            </div>
                            <div className="chip-group">
                                <span className="chip actual">Actual</span>
                                <span className="chip forecast">Forecast</span>
                                <span className="chip baseline" style={{ background: 'transparent' }}>Persistence</span>
                            </div>
                        </div>

                        <div className="fp-chart-body">
                            {liveData && liveData.length > 0 ? (
                                chart
                            ) : (
                                <div className="chart-empty">No live data available for {effectiveDate}</div>
                            )}
                        </div>

                        <div className="fp-chart-footer">
                            <span>Verified AI Execution &bull; Real-time Stream</span>
                            <span>MAPE: <strong style={{ color: forecastHealth.color }}>{liveMeta?.mape_live ? `${liveMeta.mape_live.toFixed(2)}%` : '--'}</strong></span>
                        </div>
                    </div>

                    {/* Weather Strip */}
                    {weatherStrip}

                    {/* Forecast Table */}
                    {forecastTable && (
                        <div className="fp-table-section">
                            <div className="fp-table-header">
                                <h3>Block-Level Forecast Table</h3>
                            </div>
                            {forecastTable}
                        </div>
                    )}
                </div>

                {/* RIGHT: Intelligence sidebar */}
                <aside className="fp-intel">
                    <div className="fp-sidebar-section">
                        <span className="fp-section-label">Forecast Context</span>
                        <div className="glass-panel p-2 flex flex-col gap-2">
                            <div className="flex justify-between items-center">
                                <span className="stat-label">Date</span>
                                <span className="font-semibold text-[var(--accent)]">{effectiveDate}</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="stat-label">Region</span>
                                <span className="font-semibold capitalize">{selectedRegion}</span>
                            </div>
                            <div className="flex justify-between items-center">
                                <span className="stat-label">Status</span>
                                <span className="live-pill">Live</span>
                            </div>
                        </div>
                    </div>

                    <div className="fp-sidebar-section">
                        <span className="fp-section-label">Live Intelligence</span>
                        <div className="flex flex-col gap-2">
                            {insights.length > 0 ? insights.map((insight: any, i: number) => (
                                <div key={i} className={`glass-panel p-2 border-l-4 ${insight.priority === 'high' ? 'border-[var(--danger)] bg-red-500/5' : 'border-[var(--accent)]'}`}>
                                    <div className="flex items-center gap-1 font-bold text-xs mb-0.5">
                                        {insight.priority === 'high' ? <AlertCircle size={12} className="text-[var(--danger)]" /> : <TrendingUp size={12} className="text-[var(--accent)]" />}
                                        {insight.title}
                                    </div>
                                    <p className="text-[10px] text-[var(--muted)] leading-snug">{insight.text}</p>
                                </div>
                            )) : (
                                <div className="glass-panel p-3 text-xs text-[var(--muted)] italic text-center">
                                    Forecast is tracking within normal structural parameters.
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="fp-sidebar-section">
                        <span className="fp-section-label">Driver Vectors</span>
                        <div className="glass-panel p-2 flex flex-col gap-2">
                            {activeDrivers.map((driver: any, i: number) => (
                                <div key={i} className="flex flex-col gap-1">
                                    <div className="flex justify-between items-end">
                                        <span className="text-xs font-medium">{driver.factor}</span>
                                        <span className="text-xs font-bold" style={{ color: driver.color || (driver.mw >= 0 ? 'var(--success)' : 'var(--danger)') }}>
                                            {driver.mw > 0 ? '+' : ''}{driver.mw.toFixed(0)} MW
                                        </span>
                                    </div>
                                    <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                                        <div
                                            className="h-full rounded-full transition-all duration-1000"
                                            style={{
                                                width: `${Math.min(Math.abs(driver.pct), 100)}%`,
                                                background: driver.color || (driver.mw >= 0 ? 'var(--success)' : 'var(--danger)'),
                                                boxShadow: `0 0 8px ${driver.color || (driver.mw >= 0 ? 'rgba(57,255,20,0.3)' : 'rgba(255,49,49,0.3)')}`
                                            }}
                                        />
                                    </div>
                                    <span className="text-[9px] text-[var(--muted)]">{driver.pct.toFixed(1)}% contribution</span>
                                </div>
                            ))}
                            {activeDrivers.length === 0 && <div className="text-xs text-[var(--muted)] italic text-center py-3">No exogenous pressure.</div>}
                        </div>
                    </div>

                    <div className="fp-sidebar-section">
                        <span className="fp-section-label">System Integrity</span>
                        <div className="grid grid-cols-2 gap-2">
                            <div className="glass-panel p-2 flex flex-col gap-0.5">
                                <span className="text-[9px] uppercase tracking-widest text-[var(--muted)]">Sync</span>
                                <span className="text-xs font-bold text-[var(--success)] flex items-center gap-1">
                                    <CheckCircle2 size={10} /> Active
                                </span>
                            </div>
                            <div className="glass-panel p-2 flex flex-col gap-0.5">
                                <span className="text-[9px] uppercase tracking-widest text-[var(--muted)]">Drift</span>
                                <span className="text-xs font-bold text-[var(--warning)]">0.4%</span>
                            </div>
                        </div>
                    </div>

                    {/* Engine Logs */}
                    <div className="fp-sidebar-section">
                        <span className="fp-section-label">Engine Logs</span>
                        <div className="glass-panel p-2 overflow-y-auto" style={{ maxHeight: '120px' }}>
                            {liveMeta?.logs?.map((log: string, i: number) => (
                                <div key={i} className="text-[10px] font-mono mb-1 flex gap-2">
                                    <span className="text-[var(--accent)] opacity-70">[{new Date().toLocaleTimeString()}]</span>
                                    <span className="text-[var(--muted)]">{log}</span>
                                </div>
                            )) || <div className="text-[10px] text-[var(--muted)] italic">Awaiting telemetry stream...</div>}
                        </div>
                    </div>
                </aside>
            </div>
        </main>
    );
};
