/**
 * Page-level structural primitives — every page in VidyutPragya composes
 * these so structure is consistent, premium, and easy to evolve.
 *
 * Usage:
 *   <PageShell>
 *     <PageHeader title="Forecast" subtitle="T+1 / T+2 day-ahead" actions={<...>}>
 *       <Eyebrow tone="accent">Live</Eyebrow>
 *     </PageHeader>
 *
 *     <KpiGrid>
 *       <Kpi label="Peak load" value="5,840" unit="MW" delta={+3.2} />
 *       ...
 *     </KpiGrid>
 *
 *     <ChartCard title="Block forecast" toolbar={<Toolbar>...</Toolbar>}>
 *       <YourEChartsOrCustomChart />
 *     </ChartCard>
 *
 *     <SectionGrid columns={[2, 1]}>
 *       <SectionCard title="Drivers">...</SectionCard>
 *       <SectionCard title="Decision signals">...</SectionCard>
 *     </SectionGrid>
 *   </PageShell>
 */
import React from 'react';
import { ChevronRight } from 'lucide-react';

/* ─────────────────────────────────────────────────────────────────────────── */
export function PageShell({ children, className = '', style }) {
  return (
    <div className={`vp-page ${className}`.trim()} style={style}>
      {children}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────── */
export function PageHeader({
  title,
  subtitle,
  meta,
  actions,
  children,
  size = 'md',
}) {
  return (
    <header className={`vp-page-header vp-page-header--${size}`}>
      <div className="vp-page-header__lead">
        {children && <div className="vp-page-header__chips">{children}</div>}
        <h1 className="vp-page-header__title">{title}</h1>
        {subtitle && <p className="vp-page-header__subtitle">{subtitle}</p>}
        {meta && <div className="vp-page-header__meta">{meta}</div>}
      </div>
      {actions && <div className="vp-page-header__actions">{actions}</div>}
    </header>
  );
}

/* ─────────────────────────────────────────────────────────────────────────── */
export function Eyebrow({ tone = 'default', children, ...rest }) {
  return (
    <span className={`vp-eyebrow vp-eyebrow--${tone}`} {...rest}>
      {children}
    </span>
  );
}

/* ─────────────────────────────────────────────────────────────────────────── */
export function KpiGrid({ children, columns, compact, className = '' }) {
  const cls = [
    'vp-kpi-grid',
    compact ? 'vp-kpi-grid--compact' : '',
    className,
  ].filter(Boolean).join(' ');
  const style = columns ? { gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` } : undefined;
  return <div className={cls} style={style}>{children}</div>;
}

/* ─────────────────────────────────────────────────────────────────────────── */
export function Kpi({
  label,
  value,
  unit,
  delta,
  deltaLabel,
  sub,
  tone = 'default',
  icon: Icon,
  sparkline,
  hint,
  loading,
  onClick,
  ...rest
}) {
  const isInteractive = typeof onClick === 'function';
  const Tag = isInteractive ? 'button' : 'div';
  const deltaTone =
    delta == null
      ? null
      : Number(delta) > 0
        ? 'up'
        : Number(delta) < 0
          ? 'down'
          : 'flat';

  return (
    <Tag
      className={[
        'vp-kpi',
        `vp-kpi--${tone}`,
        isInteractive ? 'vp-kpi--interactive' : '',
        loading ? 'vp-kpi--loading' : '',
      ].filter(Boolean).join(' ')}
      onClick={onClick}
      {...rest}
    >
      <div className="vp-kpi__head">
        <span className="vp-kpi__label">{label}</span>
        {Icon && <Icon size={13} className="vp-kpi__icon" />}
      </div>
      <div className="vp-kpi__body">
        <span className="vp-kpi__value">
          {loading ? <span className="vp-skeleton vp-skeleton--num" /> : value}
        </span>
        {unit && <span className="vp-kpi__unit">{unit}</span>}
      </div>
      <div className="vp-kpi__foot">
        {delta != null && (
          <span className={`vp-kpi__delta vp-kpi__delta--${deltaTone}`}>
            {Number(delta) > 0 ? '▲' : Number(delta) < 0 ? '▼' : '·'}
            {' '}
            {Math.abs(Number(delta)).toFixed(2)}
            {deltaLabel ? <span className="vp-kpi__delta-label">{deltaLabel}</span> : '%'}
          </span>
        )}
        {sub && <span className="vp-kpi__sub">{sub}</span>}
      </div>
      {sparkline && <div className="vp-kpi__spark">{sparkline}</div>}
      {hint && <span className="vp-kpi__hint" title={hint}>i</span>}
      {isInteractive && <ChevronRight size={12} className="vp-kpi__chev" />}
    </Tag>
  );
}

/* ─────────────────────────────────────────────────────────────────────────── */
export function SectionCard({
  title,
  subtitle,
  toolbar,
  badge,
  children,
  padding = 'md',
  className = '',
  bodyClassName = '',
  hoverable = false,
  ...rest
}) {
  return (
    <section
      className={[
        'vp-card',
        `vp-card--p-${padding}`,
        hoverable ? 'vp-card--hover' : '',
        className,
      ].filter(Boolean).join(' ')}
      {...rest}
    >
      {(title || subtitle || toolbar || badge) && (
        <header className="vp-card__head">
          <div className="vp-card__head-lead">
            {title && (
              <h3 className="vp-card__title">
                {title}
                {badge && <span className="vp-card__badge">{badge}</span>}
              </h3>
            )}
            {subtitle && <p className="vp-card__subtitle">{subtitle}</p>}
          </div>
          {toolbar && <div className="vp-card__toolbar">{toolbar}</div>}
        </header>
      )}
      <div className={`vp-card__body ${bodyClassName}`.trim()}>{children}</div>
    </section>
  );
}

/* ─────────────────────────────────────────────────────────────────────────── */
export function ChartCard({
  title,
  subtitle,
  toolbar,
  legend,
  height = 320,
  children,
  loading,
  empty,
  emptyMessage = 'No data available',
  className = '',
  ...rest
}) {
  return (
    <section className={`vp-chart-card ${className}`.trim()} {...rest}>
      {(title || subtitle || toolbar) && (
        <header className={`vp-chart-card__head${!title && !subtitle ? ' vp-chart-card__head--bar' : ''}`}>
          {(title || subtitle) && (
            <div className="vp-chart-card__head-lead">
              {title && <h3 className="vp-chart-card__title">{title}</h3>}
              {subtitle && <p className="vp-chart-card__subtitle">{subtitle}</p>}
            </div>
          )}
          {toolbar && (
            <div className="vp-chart-card__toolbar" style={!title && !subtitle ? { flex: 1 } : undefined}>
              {toolbar}
            </div>
          )}
        </header>
      )}
      <div className="vp-chart-card__body" style={{ height }}>
        {loading ? (
          <div className="vp-chart-card__loading">Loading chart…</div>
        ) : empty ? (
          <div className="vp-chart-card__empty">{emptyMessage}</div>
        ) : (
          children
        )}
      </div>
      {legend && <div className="vp-chart-card__legend">{legend}</div>}
    </section>
  );
}

/* ─────────────────────────────────────────────────────────────────────────── */
export function SectionGrid({ children, columns = [1], gap = 12, className = '' }) {
  const tpl = Array.isArray(columns)
    ? columns.map((c) => `${c}fr`).join(' ')
    : `repeat(${columns}, minmax(0, 1fr))`;
  return (
    <div
      className={`vp-section-grid ${className}`.trim()}
      style={{ display: 'grid', gridTemplateColumns: tpl, gap }}
    >
      {children}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────── */
export function Toolbar({ children, className = '' }) {
  return <div className={`vp-toolbar ${className}`.trim()}>{children}</div>;
}

export function Segmented({ options, value, onChange, size = 'sm' }) {
  return (
    <div className={`vp-segmented vp-segmented--${size}`} role="tablist">
      {options.map((opt) => {
        const v = typeof opt === 'string' ? opt : opt.value;
        const label = typeof opt === 'string' ? opt : opt.label;
        return (
          <button
            key={v}
            role="tab"
            aria-selected={value === v}
            className={`vp-segmented__btn ${value === v ? 'is-active' : ''}`}
            onClick={() => onChange?.(v)}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────── */
export function StatList({ items }) {
  return (
    <ul className="vp-stat-list">
      {items.map((it, i) => (
        <li key={i} className="vp-stat-list__item">
          <span className="vp-stat-list__label">{it.label}</span>
          <span className="vp-stat-list__value">
            {it.value}
            {it.unit && <span className="vp-stat-list__unit">{it.unit}</span>}
          </span>
        </li>
      ))}
    </ul>
  );
}

export function Pill({ tone = 'default', icon: Icon, children, ...rest }) {
  return (
    <span className={`vp-pill vp-pill--${tone}`} {...rest}>
      {Icon && <Icon size={11} />}
      {children}
    </span>
  );
}

export function EmptyState({ icon: Icon, title, description, action }) {
  return (
    <div className="vp-empty">
      {Icon && <Icon size={32} className="vp-empty__icon" />}
      {title && <div className="vp-empty__title">{title}</div>}
      {description && <p className="vp-empty__desc">{description}</p>}
      {action && <div className="vp-empty__action">{action}</div>}
    </div>
  );
}
