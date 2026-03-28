import React, { useMemo, useState } from 'react';
import {
    Activity,
    AlertCircle,
    BarChart3,
    CalendarDays,
    CheckCircle2,
    Clock3,
    Download,
    Gauge,
    Table2,
    TrendingUp,
    Wind,
    X,
    Zap
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

type OverlayKey = 'intelligence' | 'drivers' | 'signals' | 'logs' | 'weather' | 'table' | null;

const blockTime = (block: number) => {
    const minutes = Math.max(0, block - 1) * 15;
    const h = String(Math.floor(minutes / 60)).padStart(2, '0');
    const m = String(minutes % 60).padStart(2, '0');
    return `${h}:${m}`;
};

const titleCase = (value: string) =>
    String(value || '')
        .replace(/[_-]+/g, ' ')
        .replace(/\b\w/g, (m) => m.toUpperCase());

const tone = (value: string) => {
    if (['high', 'warning', 'critical'].includes(value)) return 'high';
    if (value === 'medium') return 'medium';
    return 'low';
};

const DetailOverlay = ({
    open,
    title,
    subtitle,
    onClose,
    children
}: {
    open: boolean;
    title: string;
    subtitle?: string;
    onClose: () => void;
    children: React.ReactNode;
}) => {
    if (!open) return null;
    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content fp-modal" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                    <div>
                        <h3>{title}</h3>
                        {subtitle && <p className="fp-modal-subtitle">{subtitle}</p>}
                    </div>
                    <button type="button" aria-label="Close overlay" onClick={onClose}>
                        <X size={18} />
                    </button>
                </div>
                <div className="modal-body">{children}</div>
            </div>
        </div>
    );
};

const KpiCard = ({
    label,
    value,
    meta,
    icon,
    className = ''
}: {
    label: string;
    value: string;
    meta: string;
    icon: React.ReactNode;
    className?: string;
}) => (
    <article className={`fp-kpi-card ${className}`.trim()}>
        <div className="fp-kpi-card__head">
            <span>{label}</span>
            <div className="fp-kpi-card__icon">{icon}</div>
        </div>
        <strong>{value}</strong>
        <p>{meta}</p>
    </article>
);

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
    const [activeOverlay, setActiveOverlay] = useState<OverlayKey>(null);
    const rows = useMemo(() => (Array.isArray(liveData) ? liveData : []), [liveData]);

    const summary = useMemo(() => {
        if (!rows.length) return null;
        const actuals = rows.filter((d: any) => d.actual_mw != null && d.actual_mw > 0);
        const lastActualRow = actuals[actuals.length - 1];
        const peakRow = rows.reduce(
            (best: any, row: any) => ((row?.forecast_mw || 0) > (best?.forecast_mw || 0) ? row : best),
            rows[0]
        );
        const energy = rows.reduce((acc: number, d: any) => acc + Number(d?.forecast_mw || 0), 0) * 0.25;
        return {
            lastActual: Number(lastActualRow?.actual_mw || 0),
            lastBlock: Number(lastActualRow?.block_number || 0),
            peak: Number(peakRow?.forecast_mw || 0),
            peakBlock: Number(peakRow?.block_number || 1),
            peakTime: peakRow?.time || blockTime(Number(peakRow?.block_number || 1)),
            energy,
            actualCoverage: Math.round((Number(lastActualRow?.block_number || 0) / 96) * 100)
        };
    }, [rows]);

    const insights = useMemo(() => {
        const raw = !liveMeta?.insights ? [] : Array.isArray(liveMeta.insights) ? liveMeta.insights : [liveMeta.insights];
        return raw
            .map((item: any, index: number) => ({
                id: `${index}-${item?.title || item?.text || 'insight'}`,
                title: item?.title || (tone(String(item?.priority || item?.type || 'low').toLowerCase()) === 'high' ? 'Attention Required' : 'Operational Insight'),
                text: item?.text || item?.message || String(item || ''),
                priority: tone(String(item?.priority || item?.type || 'low').toLowerCase())
            }))
            .filter((item: any) => item.text);
    }, [liveMeta]);

    const drivers = useMemo(() => {
        return [...(driverContributions || [])]
            .filter((d: any) => Math.abs(Number(d?.mw || 0)) > 0.5)
            .sort((a: any, b: any) => Math.abs(Number(b?.mw || 0)) - Math.abs(Number(a?.mw || 0)));
    }, [driverContributions]);

    const signals = useMemo(() => {
        const priority: Record<string, number> = { high: 3, medium: 2, low: 1 };
        return [...decisionSignals]
            .filter((signal: any) => signal?.risk_flag)
            .sort((a: any, b: any) => {
                const diff = (priority[String(b?.risk_flag || 'low')] || 0) - (priority[String(a?.risk_flag || 'low')] || 0);
                if (diff !== 0) return diff;
                return Number(b?.uncertainty_pct || 0) - Number(a?.uncertainty_pct || 0);
            });
    }, [decisionSignals]);

    const health = useMemo(() => {
        if (!rows.length) return { mape: 0, status: 'unknown', color: 'var(--muted)', consecutiveHigh: 0, needsReforecast: false };
        const actuals = rows.filter((d: any) => d.actual_mw != null && d.actual_mw > 0 && d.forecast_mw != null && d.forecast_mw > 0);
        if (actuals.length < 4) return { mape: 0, status: 'insufficient', color: 'var(--muted)', consecutiveHigh: 0, needsReforecast: false };
        const apes = actuals.map((d: any) => Math.abs(d.actual_mw - d.forecast_mw) / Math.max(d.actual_mw, 1) * 100);
        const mape = apes.reduce((a: number, b: number) => a + b, 0) / apes.length;
        let consecutive = 0;
        let maxConsecutive = 0;
        for (const ape of [...apes].reverse()) {
            if (ape > 5) {
                consecutive += 1;
                maxConsecutive = Math.max(maxConsecutive, consecutive);
            } else {
                consecutive = 0;
            }
        }
        if (mape > 5) return { mape: Math.round(mape * 100) / 100, status: 'critical', color: 'var(--danger)', consecutiveHigh: maxConsecutive, needsReforecast: maxConsecutive >= 8 };
        if (mape > 2) return { mape: Math.round(mape * 100) / 100, status: 'warning', color: 'var(--warning)', consecutiveHigh: maxConsecutive, needsReforecast: maxConsecutive >= 8 };
        return { mape: Math.round(mape * 100) / 100, status: 'good', color: 'var(--success)', consecutiveHigh: maxConsecutive, needsReforecast: maxConsecutive >= 8 };
    }, [rows]);

    const signalCount = signals.filter((signal: any) => signal.risk_flag !== 'low').length;
    const previewSignals = signals.filter((signal: any) => signal.risk_flag !== 'low').slice(0, 2);
    const previewDrivers = drivers.slice(0, 2);
    const previewLogs = Array.isArray(liveMeta?.logs) ? liveMeta.logs.slice(-2).reverse() : [];
    const maxDriverPct = Math.max(...drivers.map((driver: any) => Math.abs(Number(driver?.pct || 0))), 1);
    const avgUncertainty = signalCount
        ? signals.filter((signal: any) => signal.risk_flag !== 'low').reduce((acc: number, signal: any) => acc + Number(signal?.uncertainty_pct || 0), 0) / signalCount
        : 0;
    const driverPressure = drivers.reduce((acc: number, driver: any) => acc + Math.abs(Number(driver?.mw || 0)), 0);
    const topInsight = insights[0];

    return (
        <main className="forecast-page">
            {health.needsReforecast && (
                <div className="fp-alert-bar">
                    <div>
                        <span className="fp-alert-bar__title">Auto-Reforecast Recommended</span>
                        <p className="fp-alert-bar__text">
                            Forecast drift stayed above 5% for {health.consecutiveHigh} consecutive settled blocks.
                        </p>
                    </div>
                    <button className="fp-action-btn fp-action-btn--primary" onClick={onRefresh}>
                        <Zap size={14} />
                        Reforecast
                    </button>
                </div>
            )}

            <div className="fp-shell fp-shell--single-screen">
                <section className="fp-topbar">
                    <div className="fp-topbar__title">
                        <span className="fp-card-label">Live Forecast</span>
                        <h2>96-block operating view</h2>
                    </div>
                    <div className="fp-topbar__meta">
                        <div className="fp-meta-pill"><CalendarDays size={14} /><span>{effectiveDate}</span></div>
                        <div className="fp-meta-pill"><Activity size={14} /><span>{titleCase(selectedRegion || 'all regions')}</span></div>
                        <button className="fp-action-btn" onClick={() => setActiveOverlay('weather')}><Wind size={14} />Weather</button>
                        <button className="fp-action-btn" onClick={() => setActiveOverlay('table')}><Table2 size={14} />Table</button>
                        <button className="fp-action-btn" onClick={onRefresh}><Zap size={14} />Refresh</button>
                        {onDownload && <button className="fp-action-btn" onClick={onDownload} disabled={!canDownload}><Download size={14} />Export</button>}
                    </div>
                </section>

                <section className="fp-kpi-grid fp-kpi-grid--compact">
                    <KpiCard label="Current Actual" value={summary ? `${fmt(summary.lastActual)} MW` : '--'} meta={summary?.lastBlock ? `Block ${summary.lastBlock}` : 'Awaiting actuals'} icon={<Activity size={15} />} />
                    <KpiCard label="Forecast Peak" value={summary ? `${fmt(summary.peak)} MW` : '--'} meta={summary ? `${summary.peakTime} • B${summary.peakBlock}` : 'No peak'} icon={<TrendingUp size={15} />} className="fp-kpi-card--good" />
                    <KpiCard label="Day Energy" value={summary ? `${fmt(summary.energy)} MWh` : '--'} meta="Forecast energy" icon={<BarChart3 size={15} />} />
                    <KpiCard label="Live MAPE" value={`${fmt(liveMeta?.mape_live ?? health.mape)}%`} meta={titleCase(health.status)} icon={<Gauge size={15} />} className={`fp-kpi-card--${health.status === 'critical' ? 'danger' : health.status === 'warning' ? 'warning' : 'good'}`} />
                    <KpiCard label="Observed Blocks" value={summary ? `${summary.lastBlock}/96` : '0/96'} meta={summary ? `${summary.actualCoverage}% coverage` : 'No coverage'} icon={<Clock3 size={15} />} />
                    <KpiCard label="Risk Windows" value={`${signalCount}`} meta={signalCount ? `${fmt(avgUncertainty)}% uncertainty` : 'No active risk'} icon={<AlertCircle size={15} />} className={signalCount ? 'fp-kpi-card--warning' : 'fp-kpi-card--good'} />
                </section>

                <section className="fp-single-layout">
                    <section className="fp-chart-panel fp-chart-panel--single">
                        <div className="fp-chart-header">
                            <div>
                                <h3>Forecast Curve</h3>
                                <span>Actual, forecast, and persistence over the full day.</span>
                            </div>
                            <div className="chip-group">
                                <span className="chip actual">Actual</span>
                                <span className="chip forecast">Forecast</span>
                                <span className="chip baseline">Persistence</span>
                            </div>
                        </div>
                        <div className="fp-chart-body fp-chart-body--single">
                            {rows.length ? chart : <div className="chart-empty">No live data available for {effectiveDate}</div>}
                        </div>
                        <div className="fp-chart-footer">
                            <span>Peak {summary ? `${summary.peakTime} • ${fmt(summary.peak)} MW` : '--'}</span>
                            <span>Health <strong style={{ color: health.color }}>{health.status === 'insufficient' ? 'settling' : `${fmt(liveMeta?.mape_live ?? health.mape)}%`}</strong></span>
                        </div>
                    </section>

                    <aside className="fp-command-rail">
                        <article className="fp-side-card fp-side-card--compact">
                            <div className="fp-card-label">Context</div>
                            <div className="fp-stat-list">
                                <div className="fp-stat-row"><span>Date</span><strong>{effectiveDate}</strong></div>
                                <div className="fp-stat-row"><span>Region</span><strong>{titleCase(selectedRegion || 'all regions')}</strong></div>
                                <div className="fp-stat-row"><span>Status</span><strong className="fp-status-live">Live</strong></div>
                                <div className="fp-stat-row"><span>Coverage</span><strong>{summary ? `${summary.actualCoverage}%` : '--'}</strong></div>
                            </div>
                        </article>

                        <article className="fp-side-card fp-side-card--compact">
                            <div className="fp-card-label">Operational Brief</div>
                            <div className="fp-compact-brief">
                                <strong>{topInsight?.title || 'System nominal'}</strong>
                                <p>{topInsight?.text || 'Forecast is tracking within expected structure.'}</p>
                            </div>
                        </article>

                        <article className="fp-side-card fp-side-card--compact">
                            <div className="fp-card-label">Watchlist</div>
                            <div className="fp-watch-grid fp-watch-grid--compact">
                                <div><span>Peak</span><strong>{summary ? `${summary.peakTime} • B${summary.peakBlock}` : '--'}</strong></div>
                                <div><span>Drivers</span><strong>{fmt(driverPressure)} MW</strong></div>
                                <div><span>Risk</span><strong style={{ color: health.color }}>{titleCase(health.status)}</strong></div>
                                <div><span>Reforecast</span><strong>{health.needsReforecast ? 'Ready' : 'Stable'}</strong></div>
                            </div>
                        </article>

                        <article className="fp-side-card fp-side-card--compact">
                            <div className="fp-card-label">Quick Access</div>
                            <div className="fp-launchpad-grid fp-launchpad-grid--compact">
                                <button className="fp-detail-btn" onClick={() => setActiveOverlay('intelligence')}><span>Intel</span><strong>{insights.length}</strong></button>
                                <button className="fp-detail-btn" onClick={() => setActiveOverlay('drivers')}><span>Drivers</span><strong>{drivers.length}</strong></button>
                                <button className="fp-detail-btn" onClick={() => setActiveOverlay('signals')}><span>Signals</span><strong>{signals.length}</strong></button>
                                <button className="fp-detail-btn" onClick={() => setActiveOverlay('logs')}><span>Logs</span><strong>{Array.isArray(liveMeta?.logs) ? liveMeta.logs.length : 0}</strong></button>
                            </div>
                        </article>

                        <article className="fp-side-card fp-side-card--compact">
                            <div className="fp-side-card__head">
                                <div className="fp-card-label">Preview</div>
                                {signalCount > previewSignals.length && <button className="fp-inline-link" onClick={() => setActiveOverlay('signals')}>All</button>}
                            </div>
                            <div className="fp-preview-dual">
                                <div className="fp-preview-block">
                                    <span className="fp-preview-block__label">Signals</span>
                                    {previewSignals.length ? previewSignals.map((signal: any) => (
                                        <div key={`${signal.block}-${signal.primary_driver}`} className={`fp-signal-card fp-signal-card--${signal.risk_flag}`}>
                                            <div className="fp-signal-card__head"><span>B{signal.block}</span><strong>{signal.time || blockTime(Number(signal.block || 1))}</strong></div>
                                            <p>{signal.primary_driver || 'Structural load shift'}</p>
                                        </div>
                                    )) : <div className="fp-empty-state">No active risk windows.</div>}
                                </div>
                                <div className="fp-preview-block">
                                    <span className="fp-preview-block__label">Drivers</span>
                                    {previewDrivers.length ? previewDrivers.map((driver: any) => {
                                        const color = driver.color || (Number(driver.mw || 0) >= 0 ? 'var(--success)' : 'var(--danger)');
                                        return (
                                            <div key={driver.factor} className="fp-driver-item">
                                                <div className="fp-driver-item__head">
                                                    <span>{driver.factor}</span>
                                                    <strong style={{ color }}>{Number(driver.mw || 0) > 0 ? '+' : ''}{fmt(driver.mw)}</strong>
                                                </div>
                                                <div className="fp-driver-item__bar"><div style={{ width: `${Math.min((Math.abs(Number(driver.pct || 0)) / maxDriverPct) * 100, 100)}%`, background: color }} /></div>
                                            </div>
                                        );
                                    }) : <div className="fp-empty-state">No material driver pressure.</div>}
                                </div>
                            </div>
                        </article>

                        <article className="fp-side-card fp-side-card--compact">
                            <div className="fp-side-card__head">
                                <div className="fp-card-label">Engine Log</div>
                                {previewLogs.length > 0 && <button className="fp-inline-link" onClick={() => setActiveOverlay('logs')}>Open</button>}
                            </div>
                            <div className="fp-log-preview">
                                {previewLogs.length ? previewLogs.map((log: string, index: number) => (
                                    <div key={`${index}-${log}`} className="fp-log-line"><span>#{String(index + 1).padStart(2, '0')}</span><p>{log}</p></div>
                                )) : <div className="fp-empty-state">Awaiting telemetry events.</div>}
                            </div>
                        </article>
                    </aside>
                </section>
            </div>

            <DetailOverlay open={activeOverlay === 'intelligence'} title="Forecast Intelligence" subtitle={`${insights.length} note${insights.length === 1 ? '' : 's'}`} onClose={() => setActiveOverlay(null)}>
                <div className="fp-modal-stack">
                    {insights.length ? insights.map((insight: any) => (
                        <article key={insight.id} className={`fp-modal-note fp-modal-note--${insight.priority}`}>
                            <div className="fp-modal-note__title">{insight.priority === 'high' ? <AlertCircle size={14} /> : <TrendingUp size={14} />}<span>{insight.title}</span></div>
                            <p>{insight.text}</p>
                        </article>
                    )) : <div className="fp-empty-state">No live intelligence notes are available for this run.</div>}
                </div>
            </DetailOverlay>

            <DetailOverlay open={activeOverlay === 'drivers'} title="Driver Contribution Detail" subtitle={`${drivers.length} active driver${drivers.length === 1 ? '' : 's'}`} onClose={() => setActiveOverlay(null)}>
                <div className="fp-modal-stack">
                    {drivers.length ? drivers.map((driver: any) => {
                        const color = driver.color || (Number(driver.mw || 0) >= 0 ? 'var(--success)' : 'var(--danger)');
                        return (
                            <article key={driver.factor} className="fp-modal-driver">
                                <div className="fp-modal-driver__head">
                                    <div><strong>{driver.factor}</strong><span>{fmt(driver.pct)}% contribution share</span></div>
                                    <div style={{ color }}>{Number(driver.mw || 0) > 0 ? '+' : ''}{fmt(driver.mw)} MW</div>
                                </div>
                                <div className="fp-modal-driver__bar"><div style={{ width: `${Math.min((Math.abs(Number(driver.pct || 0)) / maxDriverPct) * 100, 100)}%`, background: color }} /></div>
                            </article>
                        );
                    }) : <div className="fp-empty-state">No active drivers above the display threshold.</div>}
                </div>
            </DetailOverlay>

            <DetailOverlay open={activeOverlay === 'signals'} title="Slot-Level Risk Signals" subtitle={`${signals.length} signal${signals.length === 1 ? '' : 's'}`} onClose={() => setActiveOverlay(null)}>
                <div className="table-wrap">
                    <table className="data-table">
                        <thead>
                            <tr><th>Block</th><th>Time</th><th>Risk</th><th>Primary Driver</th><th>Action</th><th>Uncertainty</th><th>Confidence</th></tr>
                        </thead>
                        <tbody>
                            {signals.length ? signals.map((signal: any, index: number) => (
                                <tr key={`${signal.block}-${signal.primary_driver}-${index}`}>
                                    <td>{signal.block}</td>
                                    <td>{signal.time || blockTime(Number(signal.block || 1))}</td>
                                    <td>{titleCase(signal.risk_flag || 'low')}</td>
                                    <td>{signal.primary_driver || signal.dominant_family || '—'}</td>
                                    <td>{signal.recommended_action || 'Monitor'}</td>
                                    <td>{fmt(signal.uncertainty_pct || 0)}%</td>
                                    <td>{fmt(signal.confidence || 0)}%</td>
                                </tr>
                            )) : <tr><td colSpan={7}><div className="fp-empty-state">No decision signals were produced for the current run.</div></td></tr>}
                        </tbody>
                    </table>
                </div>
            </DetailOverlay>

            <DetailOverlay open={activeOverlay === 'logs'} title="Engine Logs" subtitle={`${Array.isArray(liveMeta?.logs) ? liveMeta.logs.length : 0} event${Array.isArray(liveMeta?.logs) && liveMeta.logs.length === 1 ? '' : 's'}`} onClose={() => setActiveOverlay(null)}>
                <div className="fp-modal-stack">
                    {Array.isArray(liveMeta?.logs) && liveMeta.logs.length ? liveMeta.logs.map((log: string, index: number) => (
                        <div key={`${index}-${log}`} className="fp-modal-log"><span>Event {String(index + 1).padStart(2, '0')}</span><p>{log}</p></div>
                    )) : <div className="fp-empty-state">No engine log lines are available yet.</div>}
                </div>
            </DetailOverlay>

            <DetailOverlay open={activeOverlay === 'weather'} title="Condition Snapshot" subtitle="Moved off the main canvas to preserve a single-screen layout" onClose={() => setActiveOverlay(null)}>
                {weatherStrip ? weatherStrip : <div className="fp-empty-state">No weather strip is available.</div>}
            </DetailOverlay>

            <DetailOverlay open={activeOverlay === 'table'} title="Block-Level Forecast Table" subtitle="Detailed values, uncertainty bands, and settled actuals" onClose={() => setActiveOverlay(null)}>
                {forecastTable ? forecastTable : <div className="fp-empty-state">No forecast table is available.</div>}
            </DetailOverlay>
        </main>
    );
};
