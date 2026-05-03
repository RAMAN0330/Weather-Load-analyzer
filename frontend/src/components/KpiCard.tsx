import React from 'react'

interface KpiCardProps {
  label: string
  value: string | number
  sub?: string
  accent?: string
  icon?: React.ReactNode
  className?: string
}

export const KpiCard = ({ label, value, sub, accent, icon, className = '' }: KpiCardProps) => (
  <div className={`kpi-card ${className}`}>
    <div className="kpi-card__label">
      {icon && <span style={{ marginRight: 4 }}>{icon}</span>}
      {label}
    </div>
    <div className="kpi-card__value" style={accent ? { color: accent } : undefined}>
      {value}
    </div>
    {sub && <div className="kpi-card__sub">{sub}</div>}
  </div>
)
