import React from 'react'

export const PageSkeleton = () => (
  <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 6, padding: '6px' }}>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 6 }}>
      {[0, 1, 2, 3].map(i => (
        <div key={i} className="skeleton skeleton-kpi" />
      ))}
    </div>
    <div className="skeleton skeleton-chart" style={{ flex: 1, minHeight: 0, borderRadius: 10 }} />
  </div>
)
