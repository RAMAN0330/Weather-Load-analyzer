import React, { useMemo, useState } from 'react';
import {
    Activity,
    AlertCircle,
    CalendarDays,
    Layers,
    Sparkles,
    Table2,
    TrendingUp,
    Wind,
    X,
    Zap,
} from 'lucide-react';
import HorizonToggle from '../../components/HorizonToggle';
import { EmptyState, StatList } from '../../components/page/PagePrimitives.jsx';

/* ── colour tokens (mirrors WeatherDeepPage C object) ─────────────────────── */
const C = {
    bg:       'var(--bg)',
    card:     'var(--bg-elevated)',
    surface:  'var(--bg-surface)',
    border:   'var(--outline)',
    accent:   'var(--accent)',
    accent2:  'var(--accent2)',
    text:     'var(--text)',
    sub:      'var(--text-secondary)',
    muted:    'var(--text-muted)',
    green:    'var(--success)',
    red:      'var(--danger)',
    warn:     'var(--warning)',
    _accent:  'var(--accent)',
    _accent2: 'var(--accent2)',
    _text:    'var(--text)',
    _sub:     'var(--text-secondary)',
    _muted:   'var(--text-muted)',
    _border:  'var(--outline)',
    _card:    'var(--bg-panel)',
    _green:   'var(--success)',
    _red:     'var(--danger)',
    _warn:    'var(--warning)',
};

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
    horizon: 't1' | 't2';
    setHorizon: (h: 't1' | 't2') => void;
    t2Date?: string;
    t2Loading?: boolean;
    weatherStrip?: React.ReactNode;
    forecastTable?: React.ReactNode;
    forecastQuality?: { engine?: string; degraded?: boolean; bias_correction_applied?: boolean | null; calibration_applied?: boolean | null; spline_applied?: boolean | null; similar_day_baseline_valid?: boolean } | null;
}

type OverlayKey = 'intelligence' | 'drivers' | 'signals' | 'logs' | 'weather' | 'table' | 'similar_days' | 'context' | null;

const blockTime = (block: number) => {
    const m = Math.max(0, block - 1) * 15;
    return `${String(Math.floor(m / 60)).padStart(2,'0')}:${String(m % 60).padStart(2,'0')}`;
};

const titleCase = (v: string) =>
    String(v || '').replace(/[_-]+/g,' ').replace(/\b\w/g, m => m.toUpperCase());

const tone = (v: string) => {
    if (['high','warning','critical'].includes(v)) return 'high';
    if (v === 'medium') return 'medium';
    return 'low';
};

/* ── Metric Card — exact same style as WeatherDeepPage ──────────────────── */
function MetricCard({ label, value, unit, delta, deltaColor, sub, color, extra }: {
    label: string; value: any; unit?: string; delta?: any;
    deltaColor?: string; sub?: string; color?: string; extra?: React.ReactNode;
}) {
    const dColor = deltaColor || C._muted;
    return (
        <div style={{
            minHeight: 110,
            padding: '16px 18px',
            background: 'linear-gradient(180deg, var(--bg-elevated), var(--bg-panel))',
            borderRadius: 14,
            border: `1px solid ${C._border}`,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            gap: 8,
        }}>
            <div style={{ fontSize: 9, textTransform: 'uppercase', letterSpacing: 1.5, color: C._muted }}>
                {label}
            </div>
            <div>
                <div style={{ fontSize: 26, fontWeight: 700, color: color || C._text, lineHeight: 1.05 }}>
                    {value}
                    {unit && <span style={{ fontSize: 12, fontWeight: 500, opacity: 0.6, marginLeft: 3 }}>{unit}</span>}
                </div>
                {delta != null && (
                    <div style={{ fontSize: 10, color: dColor, marginTop: 5, fontWeight: 600 }}>
                        {delta}
                    </div>
                )}
            </div>
            {(sub || extra) && (
                <div>
                    {sub && <div style={{ fontSize: 10, color: C._muted, lineHeight: 1.45 }}>{sub}</div>}
                    {extra}
                </div>
            )}
        </div>
    );
}

/* ── Modal overlay ───────────────────────────────────────────────────────── */
function Overlay({ open, title, subtitle, onClose, children }: {
    open: boolean; title: string; subtitle?: string; onClose: () => void; children: React.ReactNode;
}) {
    if (!open) return null;
    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content fp-modal" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <div>
                        <h3>{title}</h3>
                        {subtitle && <p className="fp-modal-subtitle">{subtitle}</p>}
                    </div>
                    <button type="button" aria-label="Close" onClick={onClose}><X size={18} /></button>
                </div>
                <div className="modal-body">{children}</div>
            </div>
        </div>
    );
}

/* ── Tab pill button (matches weather page pill style) ───────────────────── */
function TabPill({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
    return (
        <button
            onClick={onClick}
            style={{
                flex: '0 0 auto',
                padding: '7px 14px',
                border: 'none',
                cursor: 'pointer',
                fontFamily: 'inherit',
                fontSize: 10,
                fontWeight: 600,
                letterSpacing: 0.5,
                borderRadius: 999,
                background: active ? `color-mix(in srgb, ${C._accent} 9%, transparent)` : 'transparent',
                color: active ? C._accent : C._muted,
                transition: 'all 0.15s',
            }}
        >
            {children}
        </button>
    );
}

/* ── Panel button (matches weather page "Open X" buttons) ────────────────── */
function PanelBtn({ onClick, children, count, countColor }: {
    onClick: () => void; children: React.ReactNode; count?: any; countColor?: string;
}) {
    return (
        <button
            onClick={onClick}
            style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 5,
                padding: '6px 11px',
                background: 'transparent',
                border: `1px solid ${C._border}`,
                borderRadius: 8,
                color: C._sub,
                fontSize: 10,
                fontWeight: 600,
                fontFamily: 'inherit',
                cursor: 'pointer',
                transition: 'all 0.15s',
                whiteSpace: 'nowrap',
            }}
            onMouseEnter={e => { (e.currentTarget as HTMLElement).style.borderColor = C._accent; (e.currentTarget as HTMLElement).style.color = C._text; }}
            onMouseLeave={e => { (e.currentTarget as HTMLElement).style.borderColor = C._border; (e.currentTarget as HTMLElement).style.color = C._sub; }}
        >
            {children}
            {count != null && (
                <span style={{
                    fontSize: 9, fontWeight: 700,
                    padding: '1px 5px', borderRadius: 8,
                    background: countColor ? `color-mix(in srgb, ${countColor} 13%, transparent)` : 'rgba(var(--overlay-rgb), 0.08)',
                    color: countColor || C._muted,
                }}>
                    {count}
                </span>
            )}
        </button>
    );
}

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════════════════ */
export default function ForecastPage({
    liveData, liveMeta, driverContributions, decisionSignals = [],
    effectiveDate, selectedRegion, onRefresh, onDownload, canDownload = true,
    fmt, chart, horizon, setHorizon, t2Date, t2Loading = false,
    weatherStrip, forecastTable, forecastQuality,
}: ForecastPageProps) {
    const [overlay, setOverlay] = useState<OverlayKey>(null);
    const rows = useMemo(() => (Array.isArray(liveData) ? liveData : []), [liveData]);

    /* ─── Derived ─────────────────────────────────────────────────────────── */
    const summary = useMemo(() => {
        if (!rows.length) return null;
        const actuals = rows.filter((d: any) => d.actual_mw != null && d.actual_mw > 0);
        const last = actuals[actuals.length - 1];
        const peak = rows.reduce((b: any, r: any) => (r?.forecast_mw||0) > (b?.forecast_mw||0) ? r : b, rows[0]);
        const energy = rows.reduce((s: number, d: any) => s + Number(d?.forecast_mw||0), 0) * 0.25;
        return {
            peak: Number(peak?.forecast_mw||0),
            peakBlock: Number(peak?.block_number||1),
            peakTime: peak?.time || blockTime(Number(peak?.block_number||1)),
            energy,
            actualCoverage: Math.round((Number(last?.block_number||0)/96)*100),
        };
    }, [rows]);

    const rampRisk = useMemo(() => {
        if (!rows.length) return { level: 'none', maxRamp: 0, atBlock: null as number|null };
        const start = rows.findIndex((r: any) => r.actual_mw == null || r.actual_mw <= 0);
        const slice = rows.slice(start === -1 ? rows.length : start, (start === -1 ? rows.length : start) + 6);
        if (slice.length < 2) return { level: 'none', maxRamp: 0, atBlock: null };
        let max = 0, atBlock: number|null = null;
        for (let i = 1; i < slice.length; i++) {
            const r = Math.abs((slice[i].forecast_mw||0)-(slice[i-1].forecast_mw||0));
            if (r > max) { max = r; atBlock = slice[i].block_number; }
        }
        return { level: max>200?'alert':max>150?'caution':'none', maxRamp: Math.round(max), atBlock };
    }, [rows]);

    const coverage = useMemo(() => {
        const w = rows.filter((r: any) => r.actual_mw != null && r.actual_mw > 0);
        return { elapsed: w.length, pct: Math.round((w.length/96)*100), energyMWh: w.reduce((s:number,r:any)=>s+Number(r.actual_mw||0)*0.25,0) };
    }, [rows]);

    const health = useMemo(() => {
        const a = rows.filter((d:any)=>d.actual_mw!=null&&d.actual_mw>0&&d.forecast_mw!=null&&d.forecast_mw>0);
        if (a.length < 4) return { mape:0, status:'insufficient', color:C._muted, consecutiveHigh:0 };
        const apes = a.map((d:any)=>Math.abs(d.actual_mw-d.forecast_mw)/Math.max(d.actual_mw,1)*100);
        const mape = apes.reduce((s:number,v:number)=>s+v,0)/apes.length;
        let cons=0, maxCons=0;
        for (const v of [...apes].reverse()) { if(v>5){cons++;maxCons=Math.max(maxCons,cons);}else{cons=0;} }
        if (mape>5) return { mape:Math.round(mape*100)/100, status:'critical', color:C._red, consecutiveHigh:maxCons };
        if (mape>2) return { mape:Math.round(mape*100)/100, status:'warning', color:C._warn, consecutiveHigh:maxCons };
        return { mape:Math.round(mape*100)/100, status:'good', color:C._green, consecutiveHigh:maxCons };
    }, [rows]);

    const drivers = useMemo(() =>
        [...(driverContributions||[])].filter((d:any)=>Math.abs(Number(d?.mw||0))>5)
            .sort((a:any,b:any)=>Math.abs(Number(b?.mw||0))-Math.abs(Number(a?.mw||0))),
    [driverContributions]);

    const signals = useMemo(() => {
        const p: Record<string,number> = {high:3,medium:2,low:1};
        return [...decisionSignals].filter((s:any)=>s?.risk_flag)
            .sort((a:any,b:any)=>(p[String(b?.risk_flag||'low')]||0)-(p[String(a?.risk_flag||'low')]||0));
    }, [decisionSignals]);

    const insights = useMemo(() => {
        const raw = !liveMeta?.insights ? [] : Array.isArray(liveMeta.insights) ? liveMeta.insights : [liveMeta.insights];
        return raw.map((item:any,i:number)=>({
            id:`${i}-${item?.title||'insight'}`,
            title: item?.title||(tone(String(item?.priority||'low').toLowerCase())==='high'?'Attention Required':'Operational Insight'),
            text: item?.text||item?.message||String(item||''),
            priority: tone(String(item?.priority||item?.type||'low').toLowerCase()),
        })).filter((x:any)=>x.text);
    }, [liveMeta]);

    const maxDriverPct = drivers.length ? Math.max(...drivers.map((d:any)=>Math.abs(Number(d?.pct||0)))) : 0;
    const driverPressure = drivers.reduce((s:number,d:any)=>s+Math.abs(Number(d?.mw||0)),0);
    const similarDaysCount = Array.isArray(liveMeta?.similar_days) ? liveMeta.similar_days.length : 0;
    const totalLogs = Array.isArray(liveMeta?.logs) ? liveMeta.logs.length : 0;

    const rampColor = rampRisk.level==='alert' ? C._red : rampRisk.level==='caution' ? C._warn : C._green;
    const healthColor = health.color;

    /* ─── Render ──────────────────────────────────────────────────────────── */
    return (
        <div className="forecast-page">

            {/* ── Reforecast alert ─────────────────────────────────────────── */}
            {health.consecutiveHigh >= 8 && (
                <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', gap:16, padding:'8px 18px', background:'color-mix(in srgb, var(--danger) 8%, transparent)', borderBottom:`1px solid color-mix(in srgb, var(--danger) 30%, transparent)`, flexShrink:0 }}>
                    <span style={{ fontSize:12, color:C._red }}>⚠ Forecast drift above 5% for {health.consecutiveHigh} consecutive blocks — reforecast recommended.</span>
                    <button className="primary-btn" onClick={onRefresh} style={{ flexShrink:0 }}><Zap size={13}/> Reforecast</button>
                </div>
            )}

            {/* ── MetricCard grid — matches weather page top section ───────── */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(6, minmax(0, 1fr))',
                gap: 10,
                padding: '12px 16px',
                flexShrink: 0,
            }}>
                <MetricCard
                    label="Peak Load"
                    value={summary ? fmt(summary.peak) : '--'}
                    unit="MW"
                    color={C._accent}
                    delta={summary ? `▲ at ${summary.peakTime} · B${summary.peakBlock}` : 'Awaiting data'}
                    deltaColor={C._accent}
                    sub="Forecast peak for the day"
                />
                <MetricCard
                    label="Daily Energy"
                    value={summary ? fmt(summary.energy) : '--'}
                    unit="MWh"
                    color={C._accent2}
                    delta={horizon==='t1' ? '96 blocks × 15 min' : 'T+2 pure forecast'}
                    sub="Total forecast energy"
                />
                <MetricCard
                    label="Actuals Coverage"
                    value={horizon==='t1' ? `${coverage.pct}` : '—'}
                    unit={horizon==='t1' ? '%' : ''}
                    color={horizon==='t1'&&coverage.pct>30 ? C._green : C._muted}
                    delta={horizon==='t1' ? `▲ ${coverage.elapsed}/96 · ${fmt(coverage.energyMWh)} MWh` : 'Pure forecast — no actuals'}
                    deltaColor={horizon==='t1'&&coverage.pct>30 ? C._green : C._muted}
                    sub="Settled blocks vs total"
                />
                {horizon==='t1' ? (
                    <MetricCard
                        label="Forecast Health"
                        value={health.status==='insufficient'||health.status==='unknown' ? '—' : fmt(liveMeta?.mape_live??health.mape)}
                        unit={health.status==='insufficient'||health.status==='unknown' ? '' : '% MAPE'}
                        color={healthColor}
                        delta={health.consecutiveHigh>0 ? `${health.consecutiveHigh} consec. high-error blocks` : '▲ Within tolerance'}
                        deltaColor={health.consecutiveHigh>0 ? C._warn : healthColor}
                        sub="Live settled blocks only"
                    />
                ) : (
                    <MetricCard
                        label="Confidence"
                        value={liveMeta?.model_confidence!=null ? `${Math.round(Number(liveMeta.model_confidence)*100)}` : '—'}
                        unit={liveMeta?.model_confidence!=null ? '%' : ''}
                        color={C._green}
                        delta="Pure forecast · no actuals"
                        sub="Model confidence band"
                    />
                )}
                {horizon==='t1' ? (
                    <MetricCard
                        label="Ramp Severity"
                        value={rampRisk.level==='alert'?'Alert':rampRisk.level==='caution'?'Caution':'Clear'}
                        color={rampColor}
                        delta={rampRisk.maxRamp>0 ? `${rampRisk.maxRamp} MW / 15m${rampRisk.atBlock?` · B${rampRisk.atBlock}`:''}` : 'Stable load profile'}
                        sub="Upcoming load ramp risk"
                        extra={<div style={{marginTop:6,display:'flex',alignItems:'center',gap:6,flexWrap:'wrap'}}><HorizonToggle horizon={horizon} setHorizon={setHorizon} t2Date={t2Date}/>{t2Loading&&<span style={{fontSize:9,color:C._muted}}>Loading…</span>}</div>}
                    />
                ) : (
                    <MetricCard
                        label="Horizon"
                        value={t2Date||'—'}
                        color={C._text}
                        delta="T+2 target date"
                        sub="Pure forecast mode"
                        extra={<div style={{marginTop:6}}><HorizonToggle horizon={horizon} setHorizon={setHorizon} t2Date={t2Date}/></div>}
                    />
                )}
                <MetricCard
                    label="Driver Pressure"
                    value={fmt(driverPressure)}
                    unit="MW Σ|"
                    color={C._accent2}
                    delta={`${drivers.length} active driver${drivers.length===1?'':'s'}`}
                    deltaColor={C._sub}
                    sub="Sum of |driver contributions|"
                />
            </div>

            {/* ── Quality banners ───────────────────────────────────────────── */}
            {forecastQuality?.degraded && (
                <div style={{ padding:'5px 18px', background:'color-mix(in srgb, var(--warning) 7%, transparent)', borderBottom:`1px solid color-mix(in srgb, var(--warning) 20%, transparent)`, fontSize:11, color:C._warn, flexShrink:0, fontFamily:"'IBM Plex Mono',monospace" }}>
                    ⚠ Statistical baseline only — AI engine unavailable
                </div>
            )}

            {/* ── Main chart card (matches weather page big card) ───────────── */}
            <div style={{
                flex: 1,
                minHeight: 0,
                margin: '0 16px 12px',
                background: C._card,
                borderRadius: 18,
                border: `1px solid ${C._border}`,
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
            }}>

                {/* Control strip — two rows matching weather page exactly */}
                <div style={{ display:'flex', flexDirection:'column', gap:0, borderBottom:`1px solid ${C._border}`, flexShrink:0 }}>

                    {/* Row 1: legend dots (left) + date/region (right) */}
                    <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:'10px 16px 0', gap:12 }}>
                        <div style={{ display:'flex', alignItems:'center', gap:16 }}>
                            {[
                                { label:'Actual',      color:C._green   },
                                { label:'Persistence', color:C._accent2 },
                                { label:'Forecast',    color:C._accent  },
                            ].map(({ label, color }) => (
                                <span key={label} style={{ display:'flex', alignItems:'center', gap:5, fontSize:10, color:C._muted, fontFamily:"'IBM Plex Mono',monospace" }}>
                                    <span style={{ width:7, height:7, borderRadius:'50%', background:color, display:'inline-block', flexShrink:0 }}/>
                                    {label}
                                </span>
                            ))}
                        </div>
                        <span style={{ fontSize:10, color:C._muted, fontFamily:"'IBM Plex Mono',monospace" }}>
                            {effectiveDate} · {titleCase(selectedRegion||'all')}
                        </span>
                    </div>

                    {/* Row 2: tab pills (left) + DETAIL PANELS buttons (right) */}
                    <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:'8px 16px 10px', gap:14 }}>

                        {/* Tab pill container — same as weather page */}
                        <div style={{ display:'inline-flex', gap:2, padding:4, background:'var(--bg-surface)', border:'1px solid var(--outline)', borderRadius:999 }}>
                            {[
                                { label:'Live',        active:true,  action:onRefresh },
                                { label:'Weather',     active:false, action:()=>setOverlay('weather') },
                                { label:'Block Table', active:false, action:()=>setOverlay('table') },
                                ...(onDownload ? [{ label:'Export', active:false, action:onDownload }] : []),
                            ].map(({ label, active, action }) => (
                                <button
                                    key={label}
                                    onClick={action}
                                    style={{
                                        padding:'5px 13px',
                                        border:'none',
                                        cursor:'pointer',
                                        fontFamily:"'IBM Plex Mono',monospace",
                                        fontSize:10,
                                        fontWeight:600,
                                        letterSpacing:0.4,
                                        borderRadius:999,
                                        background: active ? `color-mix(in srgb, ${C._accent} 9%, transparent)` : 'transparent',
                                        color: active ? C._accent : C._muted,
                                        transition:'all 0.15s',
                                        whiteSpace:'nowrap',
                                    }}
                                >
                                    {label}
                                </button>
                            ))}
                        </div>

                        {/* DETAIL PANELS — label + buttons */}
                        <div style={{ display:'flex', alignItems:'center', gap:6, flexWrap:'wrap' }}>
                            <span style={{ fontSize:9, textTransform:'uppercase', letterSpacing:1.5, color:C._muted, fontFamily:"'IBM Plex Mono',monospace", marginRight:2 }}>
                                Detail Panels
                            </span>
                            {[
                                { label:'Drivers',     count:drivers.length,       color:drivers.length>0?C._accent:undefined,  action:()=>setOverlay('drivers') },
                                { label:'Signals',     count:signals.length,       color:signals.length>0?C._warn:undefined,    action:()=>setOverlay('signals') },
                                { label:'Intelligence',count:insights.length,      color:insights.length>0?C._green:undefined,  action:()=>setOverlay('intelligence') },
                                { label:'Similar Days',count:similarDaysCount,     color:undefined,                             action:()=>setOverlay('similar_days') },
                                { label:'Engine Logs', count:totalLogs,            color:undefined,                             action:()=>setOverlay('logs') },
                                { label:'Context',     count:horizon==='t1'?`${summary?.actualCoverage??0}%`:'T+2', color:undefined, action:()=>setOverlay('context') },
                            ].map(({ label, count, color, action }) => (
                                <button
                                    key={label}
                                    onClick={action}
                                    style={{
                                        display:'inline-flex', alignItems:'center', gap:5,
                                        padding:'5px 10px',
                                        background:'transparent',
                                        border:`1px solid ${C._border}`,
                                        borderRadius:7,
                                        color:C._sub,
                                        fontSize:10,
                                        fontWeight:600,
                                        fontFamily:"'IBM Plex Mono',monospace",
                                        cursor:'pointer',
                                        whiteSpace:'nowrap',
                                    }}
                                >
                                    {label}
                                    <span style={{
                                        fontSize:9, fontWeight:700,
                                        padding:'1px 5px', borderRadius:6,
                                        background: color ? `color-mix(in srgb, ${color} 13%, transparent)` : 'rgba(var(--overlay-rgb), 0.07)',
                                        color: color || C._muted,
                                    }}>
                                        {count}
                                    </span>
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Chart body — fills remaining height */}
                <div style={{ flex:1, minHeight:0, overflow:'hidden' }}>
                    {rows.length ? chart : (
                        <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'100%', color:C._muted, fontSize:13 }}>
                            No data available for {effectiveDate}
                        </div>
                    )}
                </div>

                {/* Legend bar */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 20,
                    padding: '8px 18px',
                    borderTop: `1px solid ${C._border}`,
                    fontSize: 11,
                    color: C._muted,
                    fontFamily: "'IBM Plex Mono',monospace",
                    flexShrink: 0,
                }}>
                    <span>Peak <strong style={{color:C._text}}>{summary?`${summary.peakTime} · ${fmt(summary.peak)} MW`:'—'}</strong></span>
                    {horizon==='t1' ? (
                        <span>Health <strong style={{color:healthColor}}>{health.status==='insufficient'?'settling':`${fmt(liveMeta?.mape_live??health.mape)}% MAPE`}</strong></span>
                    ) : (
                        <span>Horizon <strong style={{color:C._accent}}>T+2 · No actuals</strong></span>
                    )}
                    {summary && <span>Energy <strong style={{color:C._text}}>{fmt(summary.energy)} MWh</strong></span>}
                </div>
            </div>

            {/* ── Overlays ─────────────────────────────────────────────────── */}
            <Overlay open={overlay==='intelligence'} title="Forecast Intelligence" subtitle={`${insights.length} note${insights.length===1?'':'s'}`} onClose={()=>setOverlay(null)}>
                <div className="fp-modal-stack">
                    {insights.length ? insights.map((ins:any)=>(
                        <article key={ins.id} className={`fp-modal-note fp-modal-note--${ins.priority}`}>
                            <div className="fp-modal-note__title">
                                {ins.priority==='high'?<AlertCircle size={14}/>:<TrendingUp size={14}/>}
                                <span>{ins.title}</span>
                            </div>
                            <p>{ins.text}</p>
                        </article>
                    )) : <EmptyState icon={Sparkles} title="No notes" description="No intelligence notes for this run."/>}
                </div>
            </Overlay>

            <Overlay open={overlay==='drivers'} title="Driver Contribution Detail" subtitle={`${drivers.length} active`} onClose={()=>setOverlay(null)}>
                <div className="fp-modal-stack">
                    {drivers.length ? drivers.map((d:any)=>{
                        const mv=Number(d.mw||0), pos=mv>=0;
                        const col=d.color||(pos?'var(--success)':'var(--accent)');
                        return (
                            <article key={d.factor} className="fp-modal-driver">
                                <div className="fp-modal-driver__head">
                                    <div><strong>{d.factor}</strong><span style={{marginLeft:8}}>{fmt(d.pct)}% share</span><span style={{fontSize:11,color:col,marginLeft:8}}>{pos?'▲ Load-adding':'▼ Load-suppressing'}</span></div>
                                    <div style={{color:col,fontWeight:700}}>{pos?'+':''}{fmt(mv)} MW</div>
                                </div>
                                <div className="fp-modal-driver__bar">
                                    <div style={{width:`${Math.min((Math.abs(Number(d.pct||0))/maxDriverPct)*100,100)}%`,background:col}}/>
                                </div>
                            </article>
                        );
                    }) : <EmptyState icon={Layers} title="No active drivers" description="Driver pressure below 5 MW threshold."/>}
                </div>
            </Overlay>

            <Overlay open={overlay==='signals'} title="Slot-Level Risk Signals" subtitle={`${signals.length} signals`} onClose={()=>setOverlay(null)}>
                <div className="table-wrap">
                    <table className="data-table">
                        <thead><tr><th>Block</th><th>Time</th><th>Risk</th><th>Primary Driver</th><th>Action</th><th>Uncertainty</th><th>Confidence</th></tr></thead>
                        <tbody>
                            {signals.length ? signals.map((s:any,i:number)=>(
                                <tr key={`${s.block}-${i}`}>
                                    <td>{s.block}</td><td>{s.time||blockTime(Number(s.block||1))}</td>
                                    <td>{titleCase(s.risk_flag||'low')}</td>
                                    <td>{s.primary_driver||s.dominant_family||'—'}</td>
                                    <td>{s.recommended_action||'Monitor'}</td>
                                    <td>{fmt(s.uncertainty_pct||0)}%</td>
                                    <td>{fmt(s.confidence||0)}%</td>
                                </tr>
                            )) : <tr><td colSpan={7}><EmptyState icon={AlertCircle} title="All clear" description="No risk signals for this run."/></td></tr>}
                        </tbody>
                    </table>
                </div>
            </Overlay>

            <Overlay open={overlay==='logs'} title="Engine Logs" subtitle={`${totalLogs} events`} onClose={()=>setOverlay(null)}>
                <div className="fp-modal-stack">
                    {Array.isArray(liveMeta?.logs)&&liveMeta.logs.length ? liveMeta.logs.map((log:string,i:number)=>(
                        <div key={i} className="fp-modal-log"><span>Event {String(i+1).padStart(2,'0')}</span><p>{log}</p></div>
                    )) : <EmptyState icon={Activity} title="No logs" description="No engine log lines available."/>}
                </div>
            </Overlay>

            <Overlay open={overlay==='context'} title="Context" subtitle="Run metadata" onClose={()=>setOverlay(null)}>
                <StatList items={[
                    {label:'Date',value:effectiveDate},{label:'Region',value:titleCase(selectedRegion||'all')},
                    {label:'Horizon',value:horizon==='t1'?'T+1 Live':'T+2 Pure'},{label:'Coverage',value:horizon==='t1'?`${summary?.actualCoverage??0}%`:'N/A'},
                    {label:'Drivers',value:drivers.length},{label:'Similar days',value:similarDaysCount},
                    {label:'Risk signals',value:signals.length},{label:'Intelligence',value:insights.length},{label:'Engine logs',value:totalLogs},
                ]}/>
            </Overlay>

            <Overlay open={overlay==='weather'} title="Weather Snapshot" subtitle="Conditions driving the forecast" onClose={()=>setOverlay(null)}>
                {weatherStrip||<EmptyState icon={Wind} title="No weather data"/>}
            </Overlay>

            <Overlay open={overlay==='table'} title="Block-Level Forecast Table" subtitle="Uncertainty bands, settled actuals" onClose={()=>setOverlay(null)}>
                {forecastTable||<EmptyState icon={Table2} title="No forecast table"/>}
            </Overlay>

            <Overlay open={overlay==='similar_days'} title="Similar Days" subtitle={`${horizon==='t2'?'T+2':'T+1'} run`} onClose={()=>setOverlay(null)}>
                <div className="table-wrap">
                    {Array.isArray(liveMeta?.similar_days)&&liveMeta.similar_days.length ? (
                        <table className="data-table">
                            <thead><tr><th>Date</th><th>Similarity</th><th>Temp Δ °C</th><th>Humidity Δ %</th><th>Rain Match</th></tr></thead>
                            <tbody>{liveMeta.similar_days.map((d:any,i:number)=>(
                                <tr key={`${d.date}-${i}`}>
                                    <td>{d.date}</td>
                                    <td>{typeof d.similarity_score==='number'?d.similarity_score.toFixed(3):'—'}</td>
                                    <td>{typeof d.temp_diff==='number'?d.temp_diff.toFixed(1):'—'}</td>
                                    <td>{typeof d.hum_diff==='number'?d.hum_diff.toFixed(1):'—'}</td>
                                    <td style={{color:d.rain_match?'var(--success)':'var(--danger)'}}>{d.rain_match?'✓':'✗'}</td>
                                </tr>
                            ))}</tbody>
                        </table>
                    ) : <EmptyState icon={CalendarDays} title="No similar days" description="No similar days data for this run."/>}
                </div>
            </Overlay>
        </div>
    );
}
