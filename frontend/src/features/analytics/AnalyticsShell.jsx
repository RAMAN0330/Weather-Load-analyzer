import React, { lazy, Suspense, useState } from 'react';
import {
  Zap, ArrowLeft, CloudSun, Activity, Layers, Search,
  Loader2, MapPin, LineChart, ChevronLeft, ChevronRight,
} from 'lucide-react';
import ThemeSwitcher from '../../components/ThemeSwitcher';

const Dashboard1Weather    = lazy(() => import('./Dashboard1Weather'));
const Dashboard2Load       = lazy(() => import('./Dashboard2Load'));
const Dashboard3Combined   = lazy(() => import('./Dashboard3Combined'));
const Dashboard4SimilarDays = lazy(() => import('./Dashboard4SimilarDays'));

const TABS = [
  { key: 'weather', label: 'Weather',        icon: CloudSun,  color: 'var(--tone-warm)' },
  { key: 'load',    label: 'Load & Forecast', icon: Activity,  color: 'var(--accent)' },
  { key: 'combined',label: 'Combined',        icon: Layers,    color: 'var(--success)' },
  { key: 'similar', label: 'Similar Days',    icon: Search,    color: 'var(--accent2)' },
];

const STATES = [
  'HARYANA','CHHATTISGARH','RAJASTHAN','ODISHA','PUNJAB','DELHI',
  'UTTAR_PRADESH','UTTARAKHAND','HIMACHAL_PRADESH','GUJARAT',
  'MADHYA_PRADESH','MAHARASHTRA','BIHAR','JHARKHAND','WEST_BENGAL',
  'TAMIL_NADU','KARNATAKA','TELANGANA','ANDHRA_PRADESH','KERALA',
];

const T = {
  bg:     'var(--bg)',
  sidebar:'rgba(var(--panel-rgb),0.92)',
  elev2:  'rgba(var(--overlay-rgb),0.07)',
  border: 'var(--outline)',
  fg:     'var(--text)',
  muted:  'var(--text-muted)',
  faint:  'var(--text-dim)',
  accent: 'var(--accent)',
};
// VidyutPragya brand mark — fixed orange tile in both themes (theming guide §4).
const BRAND = '#F07825';

function FallbackSpinner() {
  return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Loader2 size={26} style={{ color: T.accent, animation: 'spin 1s linear infinite' }} />
    </div>
  );
}

export default function AnalyticsShell({ onHome }) {
  const [tab, setTab]           = useState('weather');
  const [state, setState]       = useState('HARYANA');
  const [collapsed, setCollapsed] = useState(false);

  const SIDEBAR_W = collapsed ? 64 : 220;

  return (
    <div style={{
      display: 'flex', height: '100vh', width: '100vw',
      overflow: 'hidden', background: T.bg, color: T.fg,
      padding: 10, gap: 10, boxSizing: 'border-box',
    }}>

      {/* ── Floating Sidebar ── */}
      <aside style={{
        display: 'flex', flexDirection: 'column', flexShrink: 0,
        width: SIDEBAR_W,
        borderRadius: 18,
        border: `1px solid ${T.border}`,
        background: T.sidebar,
        backdropFilter: 'blur(20px)',
        transition: 'width 0.22s ease',
        overflow: 'hidden',
        boxShadow: '0 8px 40px rgba(var(--shadow-rgb),0.18)',
      }}>

        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 9, padding: collapsed ? '18px 0' : '18px 14px 14px', justifyContent: collapsed ? 'center' : 'flex-start' }}>
          <div style={{
            width: 30, height: 30, flexShrink: 0, borderRadius: 9,
            background: BRAND, display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: `0 0 14px color-mix(in srgb, ${BRAND} 45%, transparent)`,
          }}>
            <Zap size={14} color="#fff" />
          </div>
          {!collapsed && (
            <div style={{ overflow: 'hidden' }}>
              <div style={{ fontWeight: 700, fontSize: 13, lineHeight: 1.2, color: T.fg, whiteSpace: 'nowrap' }}>Forecast Studio</div>
              <div style={{ fontSize: 9, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase', color: T.faint, marginTop: 2 }}>Analytics</div>
            </div>
          )}
        </div>

        {/* Nav */}
        <nav style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 2, padding: '0 7px', overflowY: 'auto' }}>

          <SidebarBtn icon={<ArrowLeft size={15} />} label="Back to Home" collapsed={collapsed} onClick={onHome} color={T.muted} hoverColor={T.accent} />

          <div style={{ height: 1, background: T.border, margin: '5px 4px' }} />

          {TABS.map(({ key, label, icon: Icon, color }) => {
            const active = tab === key;
            return (
              <button
                key={key}
                onClick={() => setTab(key)}
                title={collapsed ? label : undefined}
                style={{
                  display: 'flex', alignItems: 'center',
                  gap: collapsed ? 0 : 9,
                  justifyContent: collapsed ? 'center' : 'flex-start',
                  height: 34, padding: collapsed ? 0 : '0 10px',
                  borderRadius: 10, border: 'none', cursor: 'pointer',
                  background: active ? T.elev2 : 'transparent',
                  color: active ? T.fg : T.muted,
                  fontSize: 13, fontWeight: 500,
                  transition: 'all 0.14s', position: 'relative',
                  whiteSpace: 'nowrap', overflow: 'hidden', width: '100%',
                }}
                onMouseEnter={(e) => { if (!active) { e.currentTarget.style.background = 'rgba(var(--overlay-rgb),0.04)'; e.currentTarget.style.color = T.fg; } }}
                onMouseLeave={(e) => { if (!active) { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = T.muted; } }}
              >
                {active && <span style={{ position: 'absolute', left: 0, top: '50%', transform: 'translateY(-50%)', width: 3, height: 18, borderRadius: 2, background: color }} />}
                <Icon size={14} style={{ color: active ? color : undefined, flexShrink: 0 }} />
                {!collapsed && <span>{label}</span>}
              </button>
            );
          })}

          {/* Location picker in sidebar */}
          <div style={{ height: 1, background: T.border, margin: '5px 4px' }} />

          {collapsed ? (
            <button
              title={state.replace(/_/g, ' ')}
              style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 34, borderRadius: 10, border: 'none', background: 'transparent', cursor: 'default' }}
            >
              <MapPin size={14} style={{ color: T.accent }} />
            </button>
          ) : (
            <div style={{ padding: '6px 4px 4px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                <MapPin size={12} style={{ color: T.accent }} />
                <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.09em', textTransform: 'uppercase', color: T.faint }}>Location</span>
              </div>
              <select
                value={state}
                onChange={(e) => setState(e.target.value)}
                style={{
                  width: '100%', background: 'rgba(var(--overlay-rgb),0.06)',
                  border: `1px solid ${T.border}`, borderRadius: 9,
                  color: T.fg, padding: '6px 8px', fontSize: 12,
                  fontWeight: 500, cursor: 'pointer', outline: 'none',
                }}
              >
                {STATES.map((s) => (
                  <option key={s} value={s} style={{ background: 'var(--bg-panel)' }}>{s.replace(/_/g, ' ')}</option>
                ))}
              </select>
            </div>
          )}
        </nav>

        {/* Footer — collapse toggle + Forecast link */}
        <div style={{ borderTop: `1px solid ${T.border}`, padding: '6px 7px' }}>
          <SidebarBtn icon={<LineChart size={15} />} label="Forecast" collapsed={collapsed} onClick={onHome} color={T.muted} />
          <button
            onClick={() => setCollapsed((c) => !c)}
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              width: '100%', height: 30, marginTop: 2,
              borderRadius: 9, border: 'none', background: 'transparent',
              color: T.faint, cursor: 'pointer', transition: 'all 0.14s',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(var(--overlay-rgb),0.05)'; e.currentTarget.style.color = T.fg; }}
            onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = T.faint; }}
          >
            {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
          </button>
        </div>
      </aside>

      {/* ── Main content (floating, full remaining space) ── */}
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0,
        borderRadius: 18, border: `1px solid ${T.border}`,
        background: 'rgba(var(--overlay-rgb),0.015)',
        boxShadow: '0 8px 40px rgba(var(--shadow-rgb),0.14)',
        overflow: 'hidden',
      }}>
        {/* ── Top bar — current view + theme switcher (top-right) ── */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10,
          flexShrink: 0, padding: '8px 14px', borderBottom: `1px solid ${T.border}`,
        }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: T.muted, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {TABS.find((x) => x.key === tab)?.label} · {state.replace(/_/g, ' ')}
          </span>
          <ThemeSwitcher />
        </div>

        {/* ── Dashboard content — fills space, scrolls internally ── */}
        <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <Suspense fallback={<FallbackSpinner />}>
            {tab === 'weather'  && <Dashboard1Weather state={state} />}
            {tab === 'load'     && <Dashboard2Load state={state} />}
            {tab === 'combined' && <Dashboard3Combined state={state} />}
            {tab === 'similar'  && <Dashboard4SimilarDays state={state} />}
          </Suspense>
        </div>
      </div>
    </div>
  );
}

function SidebarBtn({ icon, label, collapsed, onClick, color, hoverColor }) {
  return (
    <button
      onClick={onClick}
      title={collapsed ? label : undefined}
      style={{
        display: 'flex', alignItems: 'center',
        gap: collapsed ? 0 : 9,
        justifyContent: collapsed ? 'center' : 'flex-start',
        height: 34, padding: collapsed ? 0 : '0 10px',
        borderRadius: 10, border: 'none', cursor: 'pointer',
        background: 'transparent', color: color || T.muted,
        fontSize: 13, fontWeight: 500, transition: 'all 0.14s',
        whiteSpace: 'nowrap', overflow: 'hidden', width: '100%',
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(var(--overlay-rgb),0.05)'; e.currentTarget.style.color = hoverColor || T.fg; }}
      onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = color || T.muted; }}
    >
      <span style={{ flexShrink: 0 }}>{icon}</span>
      {!collapsed && <span>{label}</span>}
    </button>
  );
}
