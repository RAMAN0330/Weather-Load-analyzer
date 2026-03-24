import React from 'react';
import {
  Activity,
  BarChart3,
  Download,
  Gauge,
  LayoutGrid,
  Loader2,
  Play,
  RefreshCw,
  Settings,
  Sun,
  TrendingUp,
  Zap,
  Eye
} from 'lucide-react';

const CommandStrip = ({
  effectiveDate,
  selectedRegion,
  date,
  setDate,
  loading,
  onReload,
  onRefreshLive,
  onDownload,
  canDownload,
  activeView,
  setActiveView,
  forecastHealth,
  stats,
  fmt,
  liveMeta,
  onToggleContext,
  contextOpen,
  onViewData
}) => {
  const healthLabel = forecastHealth?.status === 'good' ? 'GREEN'
    : forecastHealth?.status === 'warning' ? 'YELLOW'
    : forecastHealth?.status === 'critical' ? 'RED' : '--';
  const healthColor = forecastHealth?.color || 'var(--muted)';

  const views = [
    { key: 'load_analysis', label: 'Load', icon: Activity },
    { key: 'weather_analysis', label: 'Weather', icon: Sun },
    { key: 'optimizer', label: 'Optimizer', icon: Gauge },
    { key: 'simulator', label: 'Simulator', icon: Play },
    { key: 'analysis', label: 'Analysis', icon: BarChart3 },
    { key: 'forecast', label: 'Forecast', icon: LayoutGrid },
    { key: 'monitor', label: 'Monitor', icon: TrendingUp },
    { key: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <div className="cs-header-row">
      {/* ─── Brand (Left) ─── */}
      <div className="cs-float cs-float--brand">
        <span className="status-dot status-dot--green status-dot--active" />
        <Zap size={15} style={{ color: 'var(--accent)' }} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
          <span className="cs-title">VidyutPragya</span>
          <span className="cs-region">{selectedRegion || 'GRID CONTROL'}</span>
        </div>
      </div>

      {/* ─── Navigation (Center) ─── */}
      <nav className="cs-float cs-float--nav">
        {views.map(v => {
          const Icon = v.icon;
          return (
            <button
              key={v.key}
              className={`cs-nav-btn ${activeView === v.key ? 'active' : ''}`}
              onClick={() => setActiveView(v.key)}
              title={v.label}
            >
              <Icon size={14} />
              <span>{v.label}</span>
            </button>
          );
        })}
      </nav>

      {/* ─── Actions (Right) ─── */}
      <div className="cs-float cs-float--actions">
        <input
          type="date"
          className="cs-date-input"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          aria-label="Select date"
        />
        <button className="cs-action-btn" onClick={onRefreshLive} title="Refresh live forecast">
          {loading ? <Loader2 size={14} className="spin" /> : <RefreshCw size={14} />}
        </button>
        {onDownload && (
          <button className="cs-action-btn" onClick={onDownload} disabled={!canDownload} title="Download CSV">
            <Download size={14} />
          </button>
        )}
        <button className="cs-action-btn" onClick={onViewData} title="View raw data">
          <Eye size={14} />
        </button>
        <button className={`cs-action-btn ${contextOpen ? 'active' : ''}`} onClick={onToggleContext} title="Toggle context panel">
          <Activity size={14} />
        </button>
      </div>
    </div>
  );
};

export default CommandStrip;
