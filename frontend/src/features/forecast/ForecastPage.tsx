import React, { useMemo, useState } from 'react';
import {
    Activity,
    AlertCircle,
    BarChart3,
    CalendarDays,
    Clock3,
    Download,
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
    t2Chart?: React.ReactNode;
    t2LiveData?: any[];
    t2Meta?: any;
    weatherStrip?: React.ReactNode;
    forecastTable?: React.ReactNode;
}

type OverlayKey = 'intelligence' | 'drivers' | 'signals' | 'logs' | 'weather' | 'table' | 'preview' | null;

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
    eyebrow,
    title,
    value,
    unit,
    tone = '#ECEEF3',
    detail,
    footer,
    icon
}: {
    eyebrow: string;
    title: string;
    value: string;
    unit?: string;
    tone?: string;
    detail?: string;
    footer?: string;
    icon?: React.ReactNode;
}) => (
    <div
        className="fp-kpi-card"
        style={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            padding: '16px 18px',
            gap: '8px',
            minHeight: '132px',
            background: 'linear-gradient(180deg, rgba(30, 29, 35, 0.98), rgba(22, 22, 27, 0.98))',
            borderRadius: '14px',
            border: '1px solid #2A292F',
            boxShadow: '0 18px 40px rgba(0, 0, 0, 0.18)',
            position: 'relative'
        }}
    >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontSize: '8px', letterSpacing: '1.1px', textTransform: 'uppercase', color: '#6B7186' }}>{eyebrow}</div>
            {icon && <div style={{ opacity: 0.5, color: '#A0A5B8' }}>{icon}</div>}
        </div>
        <div>
            <div style={{ fontSize: '13px', fontWeight: 600, color: '#A0A5B8', marginBottom: '8px' }}>{title}</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', flexWrap: 'wrap' }}>
                <span style={{ fontSize: '30px', lineHeight: 1, fontWeight: 700, color: tone }}>{value}</span>
                {unit ? <span style={{ fontSize: '11px', color: '#6B7186' }}>{unit}</span> : null}
            </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {detail ? <div style={{ fontSize: '10px', color: '#ECEEF3' }}>{detail}</div> : null}
            {footer ? <div style={{ fontSize: '9px', color: '#6B7186' }}>{footer}</div> : null}
        </div>
    </div>
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
    t2Chart,
    t2LiveData,
    t2Meta,
    weatherStrip,
    forecastTable
}) => {
    const [activeOverlay, setActiveOverlay] = useState<OverlayKey>(null);
    const [forecastHorizon, setForecastHorizon] = useState<'t1' | 't2'>('t1');
    const rows = useMemo(() => (Array.isArray(liveData) ? liveData : []), [liveData]);
    const t2Rows = useMemo(() => (Array.isArray(t2LiveData) ? t2LiveData : []), [t2LiveData]);

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

    const t2Summary = useMemo(() => {
        if (!t2Rows.length) return null;
        const peakRow = t2Rows.reduce(
            (best: any, row: any) => ((row?.forecast_mw || 0) > (best?.forecast_mw || 0) ? row : best),
            t2Rows[0]
        );
        const energy = t2Rows.reduce((acc: number, d: any) => acc + Number(d?.forecast_mw || 0), 0) * 0.25;
        return {
            peak: Number(peakRow?.forecast_mw || 0),
            peakBlock: Number(peakRow?.block_number || 1),
            peakTime: peakRow?.time || blockTime(Number(peakRow?.block_number || 1)),
            energy,
        };
    }, [t2Rows]);

    const activeSummary = forecastHorizon === 't2' ? t2Summary : summary;
    const activeMeta = forecastHorizon === 't2' ? t2Meta : liveMeta;

    // Ramp Risk: flag if max forecast ramp in next 6 blocks exceeds 150 MW (state ramp limit proxy)
    const rampRisk = useMemo(() => {
        if (!rows.length) return { level: 'none', maxRamp: 0, atBlock: null };
        const nowBlock = rows.findIndex((r: any) => r.actual_mw == null || r.actual_mw <= 0);
        const lookAhead = rows.slice(Math.max(0, nowBlock), nowBlock + 6);
        if (lookAhead.length < 2) return { level: 'none', maxRamp: 0, atBlock: null };
        let maxRamp = 0;
        let atBlock: number | null = null;
        for (let i = 1; i < lookAhead.length; i++) {
            const ramp = Math.abs((lookAhead[i].forecast_mw || 0) - (lookAhead[i - 1].forecast_mw || 0));
            if (ramp > maxRamp) { maxRamp = ramp; atBlock = lookAhead[i].block_number; }
        }
        const level = maxRamp > 200 ? 'alert' : maxRamp > 150 ? 'caution' : 'none';
        return { level, maxRamp: Math.round(maxRamp), atBlock };
    }, [rows]);

    // Actuals coverage: elapsed blocks with actual readings
    const actualsCoverage = useMemo(() => {
        if (!rows.length) return { elapsed: 0, total: 96, pct: 0, energyMWh: 0 };
        const withActuals = rows.filter((r: any) => r.actual_mw != null && r.actual_mw > 0);
        const elapsed = withActuals.length;
        const energyMWh = withActuals.reduce((acc: number, r: any) => acc + Number(r.actual_mw || 0) * 0.25, 0);
        return { elapsed, total: 96, pct: Math.round((elapsed / 96) * 100), energyMWh };
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
            .filter((d: any) => Math.abs(Number(d?.mw || 0)) > 5)  // threshold raised from 0.5 to 5 MW to remove noise
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
                <section className="fp-kpi-grid" style={{ 
                    display: 'grid', 
                    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
                    gap: '12px', 
                    marginBottom: '20px' 
                }}>
                    <KpiCard 
                        eyebrow="Peak posture"
                        title="Peak Load" 
                        value={activeSummary ? fmt(activeSummary.peak) : '--'} 
                        unit="MW"
                        tone="#F07825"
                        detail={activeSummary ? `Expected at ${activeSummary.peakTime}` : 'No peak data'}
                        footer={activeSummary ? `±1 blk · Block ${activeSummary.peakBlock}` : '--'}
                        icon={<TrendingUp size={15} />} 
                    />
                    <KpiCard 
                        eyebrow="Energy balance"
                        title="Daily Energy" 
                        value={activeSummary ? fmt(activeSummary.energy) : '--'} 
                        unit="MWh"
                        tone="#ECEEF3"
                        detail={forecastHorizon === 't1' ? 'Σ 96 blocks × 15min' : 'T+2 pure forecast'}
                        footer="Total daily throughput" 
                        icon={<BarChart3 size={15} />} 
                    />
                    <KpiCard
                        eyebrow="Live tracker"
                        title="Actuals Coverage"
                        value={forecastHorizon === 't1' ? `${actualsCoverage.pct}%` : 'N/A'}
                        tone="#34D399"
                        detail={forecastHorizon === 't1' ? `${fmt(actualsCoverage.energyMWh)} MWh accumulated` : 'T+2 has no actuals'}
                        footer={forecastHorizon === 't1' ? `${actualsCoverage.elapsed}/96 blocks covered` : 'Pure forecast mode'}
                        icon={<Activity size={15} />}
                    />
                    <KpiCard
                        eyebrow="Operational risk"
                        title="Ramp Severity"
                        value={rampRisk.level === 'alert' ? 'Alert' : rampRisk.level === 'caution' ? 'Caution' : 'Clear'}
                        tone={rampRisk.level === 'alert' ? '#F87171' : rampRisk.level === 'caution' ? '#FBBF24' : '#34D399'}
                        detail={rampRisk.maxRamp > 0 ? `${rampRisk.maxRamp} MW/15m swing` : 'Stable load profile'}
                        footer={rampRisk.atBlock ? `Critical window at B${rampRisk.atBlock}` : 'No immediate risks'}
                        icon={<AlertCircle size={15} />}
                    />
                </section>

                <section style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
                    <div style={{ display: 'flex', gap: '8px' }}>
                        <div className="fp-meta-pill"><CalendarDays size={14} /><span>{effectiveDate}</span></div>
                        <div className="fp-meta-pill"><Activity size={14} /><span>{titleCase(selectedRegion || 'all regions')}</span></div>
                    </div>
                    <div style={{ display: 'flex', gap: '8px' }}>
                        <button className="fp-action-btn" onClick={() => setActiveOverlay('weather')}><Wind size={14} />Weather</button>
                        <button className="fp-action-btn" onClick={() => setActiveOverlay('table')}><Table2 size={14} />Table</button>
                        <button className="fp-action-btn" onClick={onRefresh}><Zap size={14} />Refresh</button>
                        {onDownload && <button className="fp-action-btn" onClick={onDownload} disabled={!canDownload}><Download size={14} />Export</button>}
                    </div>
                </section>

                <section className="fp-single-layout">
                    <section className="fp-chart-panel fp-chart-panel--single">
                        <div className="fp-chart-header">
                            <div>
                                <h3>Forecast Curve</h3>
                                <span>
                                    {forecastHorizon === 't1'
                                        ? 'Actual, forecast, and persistence over the full day.'
                                        : 'T+2 pure forecast — no actuals available.'}
                                </span>
                            </div>
                            <div className="chip-group" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <div className="fp-horizon-tabs" style={{ display: 'inline-flex', gap: '4px', padding: '5px', background: '#141419', border: '1px solid #2A292F', borderRadius: '999px', marginRight: '16px' }}>
                                    <button
                                        style={{ 
                                            flex: '0 0 auto',
                                            padding: '6px 14px',
                                            fontSize: '10px',
                                            fontWeight: 600,
                                            letterSpacing: '0.3px',
                                            cursor: 'pointer',
                                            border: '1px solid transparent',
                                            fontFamily: 'inherit',
                                            borderRadius: '999px',
                                            background: forecastHorizon === 't1' ? '#F0782518' : 'transparent',
                                            color: forecastHorizon === 't1' ? '#F07825' : '#A0A5B8',
                                            transition: 'all 0.15s ease'
                                        }}
                                        onClick={() => setForecastHorizon('t1')}
                                        title="T+1: Tomorrow (with actuals as they settle)"
                                    >
                                        T+1
                                    </button>
                                    <button
                                        style={{ 
                                            flex: '0 0 auto',
                                            padding: '6px 14px',
                                            fontSize: '10px',
                                            fontWeight: 600,
                                            letterSpacing: '0.3px',
                                            cursor: t2Chart ? 'pointer' : 'not-allowed',
                                            border: '1px solid transparent',
                                            fontFamily: 'inherit',
                                            borderRadius: '999px',
                                            background: forecastHorizon === 't2' ? '#F0782518' : 'transparent',
                                            color: forecastHorizon === 't2' ? '#F07825' : '#A0A5B8',
                                            opacity: t2Chart ? 1 : 0.5,
                                            transition: 'all 0.15s ease'
                                        }}
                                        onClick={() => setForecastHorizon('t2')}
                                        disabled={!t2Chart}
                                        title="T+2: Day after tomorrow"
                                    >
                                        T+2
                                    </button>
                                </div>
                                {forecastHorizon === 't1' && (
                                    <>
                                        <span className="chip actual">Actual</span>
                                        <span className="chip forecast">Forecast</span>
                                        <span className="chip baseline">Persistence</span>
                                    </>
                                )}
                                {forecastHorizon === 't2' && (
                                    <>
                                        <span className="chip forecast">T+2 Forecast</span>
                                        <span className="chip baseline">Baseline</span>
                                    </>
                                )}
                            </div>
                        </div>
                        <div className="fp-chart-body fp-chart-body--single">
                            {forecastHorizon === 't1'
                                ? (rows.length ? chart : <div className="chart-empty">No live data available for {effectiveDate}</div>)
                                : (t2Chart || <div className="chart-empty">T+2 forecast not loaded. Click Refresh to generate.</div>)
                            }
                        </div>
                        <div className="fp-chart-footer">
                            <span>Peak {activeSummary ? `${activeSummary.peakTime} • ${fmt(activeSummary.peak)} MW` : '--'}</span>
                            {forecastHorizon === 't1'
                                ? <span>Health <strong style={{ color: health.color }}>{health.status === 'insufficient' ? 'settling' : `${fmt(liveMeta?.mape_live ?? health.mape)}%`}</strong></span>
                                : <span>Horizon <strong style={{ color: 'var(--accent)' }}>T+2 · No actuals</strong></span>
                            }
                        </div>
                    </section>

                    <aside className="fp-command-rail" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                        <article className="fp-side-card fp-side-card--compact">
                            <div className="fp-card-label">Context</div>
                            <div className="fp-stat-list">
                                <div className="fp-stat-row"><span>Date</span><strong>{effectiveDate}</strong></div>
                                <div className="fp-stat-row"><span>Region</span><strong>{titleCase(selectedRegion || 'all regions')}</strong></div>
                                <div className="fp-stat-row"><span>Horizon</span><strong className="fp-status-live">{forecastHorizon === 't1' ? 'T+1 Live' : 'T+2 Pure'}</strong></div>
                                <div className="fp-stat-row"><span>Coverage</span><strong>{forecastHorizon === 't1' ? (summary ? `${summary.actualCoverage}%` : '--') : 'N/A (T+2)'}</strong></div>
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
                                <button className="fp-detail-btn" onClick={() => setActiveOverlay('preview')}><span>Preview</span><strong>{previewSignals.length + previewDrivers.length}</strong></button>
                            </div>
                        </article>

                        <article className="fp-side-card fp-side-card--compact" style={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                            <div className="fp-side-card__head">
                                <div className="fp-card-label">Engine Log</div>
                                {previewLogs.length > 0 && <button className="fp-inline-link" onClick={() => setActiveOverlay('logs')}>Open</button>}
                            </div>
                            <div className="fp-log-preview" style={{ flexGrow: 1 }}>
                                {previewLogs.length ? previewLogs.map((log: string, index: number) => (
                                    <div key={`${index}-${log}`} className="fp-log-line"><span>#{String(index + 1).padStart(2, '0')}</span><p>{log}</p></div>
                                )) : <div className="fp-empty-state">Awaiting telemetry events.</div>}
                            </div>
                        </article>
                    </aside>
                </section>
            </div>

            <DetailOverlay open={activeOverlay === 'preview'} title="Preview" subtitle="Top signals and driver contributions" onClose={() => setActiveOverlay(null)}>
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
            </DetailOverlay>

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
                        const mwVal = Number(driver.mw || 0);
                        const isPositive = mwVal >= 0;
                        const color = driver.color || (isPositive ? 'var(--success)' : 'var(--accent)');
                        const dirLabel = isPositive ? '▲ Load-adding' : '▼ Load-suppressing';
                        return (
                            <article key={driver.factor} className="fp-modal-driver">
                                <div className="fp-modal-driver__head">
                                    <div>
                                        <strong>{driver.factor}</strong>
                                        <span>{fmt(driver.pct)}% share</span>
                                        <span style={{ fontSize: '11px', color, marginLeft: '6px' }}>{dirLabel}</span>
                                    </div>
                                    <div style={{ color, fontWeight: 600 }}>{mwVal > 0 ? '+' : ''}{fmt(mwVal)} MW</div>
                                </div>
                                <div className="fp-modal-driver__bar"><div style={{ width: `${Math.min((Math.abs(Number(driver.pct || 0)) / maxDriverPct) * 100, 100)}%`, background: color }} /></div>
                            </article>
                        );
                    }) : <div className="fp-empty-state">No active drivers above the 5 MW display threshold.</div>}
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
