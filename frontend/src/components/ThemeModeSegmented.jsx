import React from 'react';
import { Monitor, Moon, Sun } from 'lucide-react';
import { useResolvedTheme, useThemeMode } from '../features/studio/theme';

const OPTIONS = [
  { value: 'light', label: 'Light', Icon: Sun },
  { value: 'dark', label: 'Dark', Icon: Moon },
  { value: 'system', label: 'System', Icon: Monitor },
];

/**
 * Always-visible, labelled Light / Dark (One Dark Pro) / System selector.
 * Used where the theme must be obvious before sign-in (login, landing).
 */
export default function ThemeModeSegmented({ showLabel = true, style }) {
  const [mode, setMode] = useThemeMode();
  const resolved = useResolvedTheme();

  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 10, ...style }}>
      {showLabel && (
        <span
          id="vp-appearance-label"
          style={{
            fontSize: 11,
            fontWeight: 700,
            letterSpacing: 0.6,
            textTransform: 'uppercase',
            color: 'var(--text-muted)',
          }}
        >
          Appearance
        </span>
      )}
      <div
        role="radiogroup"
        aria-labelledby={showLabel ? 'vp-appearance-label' : undefined}
        aria-label={showLabel ? undefined : 'Appearance'}
        style={{
          display: 'inline-flex',
          gap: 2,
          padding: 3,
          borderRadius: 999,
          background: 'var(--bg-surface)',
          border: '1px solid var(--outline)',
          boxShadow: '0 4px 14px rgba(var(--shadow-rgb), 0.08)',
        }}
      >
        {OPTIONS.map(({ value, label, Icon }) => {
          const active = mode === value;
          return (
            <button
              key={value}
              type="button"
              role="radio"
              aria-checked={active}
              title={value === 'system' ? `Follow system (${resolved})` : `${label} theme`}
              onClick={() => setMode(value)}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                padding: '6px 12px',
                borderRadius: 999,
                border: 'none',
                cursor: 'pointer',
                fontFamily: 'inherit',
                fontSize: 12,
                fontWeight: 700,
                background: active ? 'var(--accent)' : 'transparent',
                color: active ? 'var(--accent-fg)' : 'var(--text-secondary)',
                transition: 'background 150ms ease, color 150ms ease',
              }}
            >
              <Icon size={14} aria-hidden="true" />
              {label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
