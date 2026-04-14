import React from 'react';

/**
 * Reusable T+1 / T+2 horizon toggle pill.
 * Props:
 *   horizon    : 't1' | 't2'
 *   setHorizon : setter
 *   t2Date     : string like '2026-04-15' (disables T+2 button when absent)
 *   style      : optional container style overrides
 */
export default function HorizonToggle({ horizon, setHorizon, t2Date, style }) {
  const t2Disabled = !t2Date;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 3, background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.10)', borderRadius: 999, padding: '3px 4px', ...style }}>
      {[
        { key: 't1', label: 'T+1' },
        { key: 't2', label: t2Date ? `T+2 · ${t2Date}` : 'T+2' },
      ].map(({ key, label }) => {
        const active = horizon === key;
        const disabled = key === 't2' && t2Disabled;
        return (
          <button
            key={key}
            onClick={() => !disabled && setHorizon(key)}
            title={key === 't2' && t2Disabled ? 'T+2 data not yet available' : label}
            style={{
              padding: '4px 12px',
              borderRadius: 999,
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: 0.3,
              cursor: disabled ? 'not-allowed' : 'pointer',
              border: 'none',
              fontFamily: 'inherit',
              background: active ? '#F07825' : 'transparent',
              color: active ? '#fff' : disabled ? '#4A4D5E' : '#A0A5B8',
              opacity: disabled ? 0.45 : 1,
              transition: 'all 0.15s',
              whiteSpace: 'nowrap',
            }}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}
