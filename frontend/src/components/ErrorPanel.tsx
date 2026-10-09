import React, { useEffect, useState } from 'react'
import { AlertTriangle, RefreshCw, X, WifiOff, Database, Cloud } from 'lucide-react'

export type ErrorKind = 'network' | 'data' | 'weather' | 'server' | 'unknown'

export interface ErrorPanelProps {
  open: boolean
  title?: string
  message: string
  kind?: ErrorKind
  retryable?: boolean
  onRetry?: () => void | Promise<void>
  onDismiss?: () => void
  details?: string
}

const ICON_FOR_KIND: Record<ErrorKind, React.ReactNode> = {
  network: <WifiOff size={22} strokeWidth={2.2} />,
  data:    <Database size={22} strokeWidth={2.2} />,
  weather: <Cloud size={22} strokeWidth={2.2} />,
  server:  <AlertTriangle size={22} strokeWidth={2.2} />,
  unknown: <AlertTriangle size={22} strokeWidth={2.2} />,
}

const LABEL_FOR_KIND: Record<ErrorKind, string> = {
  network: 'Connection Issue',
  data:    'Data Unavailable',
  weather: 'Weather Data Missing',
  server:  'Server Error',
  unknown: 'Something Went Wrong',
}

export const ErrorPanel: React.FC<ErrorPanelProps> = ({
  open,
  title,
  message,
  kind = 'unknown',
  retryable = true,
  onRetry,
  onDismiss,
  details,
}) => {
  const [retrying, setRetrying] = useState(false)
  const [showDetails, setShowDetails] = useState(false)

  useEffect(() => {
    if (!open) {
      setRetrying(false)
      setShowDetails(false)
    }
  }, [open])

  if (!open) return null

  const handleRetry = async () => {
    if (!onRetry || retrying) return
    setRetrying(true)
    try {
      await onRetry()
    } finally {
      setRetrying(false)
    }
  }

  const accent = kind === 'weather' ? 'var(--info)' : kind === 'network' ? 'var(--warning)' : 'var(--danger)'
  const label = title || LABEL_FOR_KIND[kind]
  const icon = ICON_FOR_KIND[kind]

  return (
    <div
      onClick={onDismiss}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(var(--shadow-rgb), 0.6)',
        backdropFilter: 'blur(6px)',
        WebkitBackdropFilter: 'blur(6px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
        animation: 'errPanelFadeIn 180ms ease',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 'min(460px, 92vw)',
          background: 'linear-gradient(180deg, var(--bg-elevated) 0%, var(--bg-panel) 100%)',
          border: `1px solid color-mix(in srgb, ${accent} 20%, transparent)`,
          borderRadius: 14,
          padding: 0,
          boxShadow: `0 20px 60px rgba(var(--shadow-rgb), 0.5), 0 0 0 1px color-mix(in srgb, ${accent} 8%, transparent), 0 0 40px color-mix(in srgb, ${accent} 13%, transparent)`,
          animation: 'errPanelSlideIn 220ms cubic-bezier(0.2, 0.9, 0.3, 1.1)',
          fontFamily: "'Manrope', system-ui, sans-serif",
          overflow: 'hidden',
        }}
      >
        {/* Accent bar */}
        <div style={{ height: 3, background: `linear-gradient(90deg, transparent, ${accent}, transparent)` }} />

        {/* Header */}
        <div style={{ padding: '20px 24px 16px', display: 'flex', alignItems: 'flex-start', gap: 14 }}>
          <div
            style={{
              width: 42,
              height: 42,
              borderRadius: 10,
              background: `color-mix(in srgb, ${accent} 8%, transparent)`,
              border: `1px solid color-mix(in srgb, ${accent} 20%, transparent)`,
              color: accent,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
              animation: 'errPanelIconPulse 2.4s ease-in-out infinite',
            }}
          >
            {icon}
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
              <div
                style={{
                  fontSize: 15,
                  fontWeight: 700,
                  color: 'var(--text)',
                  letterSpacing: '-0.2px',
                  lineHeight: 1.2,
                }}
              >
                {label}
              </div>
              {onDismiss && (
                <button
                  type="button"
                  onClick={onDismiss}
                  aria-label="Close"
                  style={{
                    width: 26,
                    height: 26,
                    borderRadius: 6,
                    background: 'transparent',
                    border: '1px solid rgba(var(--overlay-rgb), 0.08)',
                    color: 'var(--text-muted)',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'all 0.15s ease',
                    flexShrink: 0,
                  }}
                  onMouseEnter={(e) => {
                    ;(e.currentTarget as HTMLElement).style.background = 'rgba(var(--overlay-rgb), 0.06)'
                    ;(e.currentTarget as HTMLElement).style.color = 'var(--text)'
                  }}
                  onMouseLeave={(e) => {
                    ;(e.currentTarget as HTMLElement).style.background = 'transparent'
                    ;(e.currentTarget as HTMLElement).style.color = 'var(--text-muted)'
                  }}
                >
                  <X size={14} />
                </button>
              )}
            </div>
            <div
              style={{
                fontSize: 12,
                color: 'var(--text-secondary)',
                marginTop: 6,
                lineHeight: 1.55,
                fontFamily: "'IBM Plex Mono', monospace",
              }}
            >
              {message}
            </div>
          </div>
        </div>

        {/* Info banner */}
        <div
          style={{
            margin: '0 24px',
            padding: '10px 12px',
            background: `color-mix(in srgb, ${accent} 5%, transparent)`,
            border: `1px solid color-mix(in srgb, ${accent} 13%, transparent)`,
            borderRadius: 8,
            fontSize: 11,
            color: accent,
            lineHeight: 1.5,
            display: 'flex',
            alignItems: 'flex-start',
            gap: 8,
          }}
        >
          <div style={{ fontWeight: 700, letterSpacing: '0.04em', textTransform: 'uppercase', fontSize: 10 }}>
            {retryable ? 'Retry may resolve this' : 'Action needed'}
          </div>
        </div>

        {/* Details toggle */}
        {details && (
          <div style={{ padding: '12px 24px 0' }}>
            <button
              type="button"
              onClick={() => setShowDetails((v) => !v)}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--text-muted)',
                fontSize: 11,
                fontFamily: "'IBM Plex Mono', monospace",
                cursor: 'pointer',
                padding: 0,
                letterSpacing: '0.04em',
              }}
            >
              {showDetails ? '▼ Hide' : '▶ Show'} technical details
            </button>
            {showDetails && (
              <pre
                style={{
                  marginTop: 8,
                  padding: '8px 10px',
                  background: 'var(--bg-surface)',
                  border: '1px solid rgba(var(--overlay-rgb), 0.05)',
                  borderRadius: 6,
                  fontSize: 10,
                  color: 'var(--text-muted)',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  maxHeight: 120,
                  overflowY: 'auto',
                  fontFamily: "'IBM Plex Mono', monospace",
                }}
              >
                {details}
              </pre>
            )}
          </div>
        )}

        {/* Actions */}
        <div
          style={{
            padding: '16px 24px 20px',
            display: 'flex',
            gap: 8,
            justifyContent: 'flex-end',
            alignItems: 'center',
          }}
        >
          {onDismiss && (
            <button
              type="button"
              onClick={onDismiss}
              style={{
                padding: '8px 16px',
                borderRadius: 8,
                border: '1px solid var(--outline)',
                background: 'transparent',
                color: 'var(--text-secondary)',
                fontSize: 12,
                fontWeight: 600,
                cursor: 'pointer',
                fontFamily: "'Manrope', sans-serif",
                transition: 'all 0.15s ease',
              }}
              onMouseEnter={(e) => {
                ;(e.currentTarget as HTMLElement).style.background = 'rgba(var(--overlay-rgb), 0.04)'
              }}
              onMouseLeave={(e) => {
                ;(e.currentTarget as HTMLElement).style.background = 'transparent'
              }}
            >
              Dismiss
            </button>
          )}
          {retryable && onRetry && (
            <button
              type="button"
              onClick={handleRetry}
              disabled={retrying}
              style={{
                padding: '8px 18px',
                borderRadius: 8,
                border: 'none',
                background: retrying ? `color-mix(in srgb, ${accent} 40%, transparent)` : accent,
                color: 'var(--accent-fg)',
                fontSize: 12,
                fontWeight: 700,
                cursor: retrying ? 'wait' : 'pointer',
                fontFamily: "'Manrope', sans-serif",
                transition: 'all 0.15s ease',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                boxShadow: `0 4px 14px color-mix(in srgb, ${accent} 27%, transparent)`,
              }}
              onMouseEnter={(e) => {
                if (!retrying) (e.currentTarget as HTMLElement).style.transform = 'translateY(-1px)'
              }}
              onMouseLeave={(e) => {
                ;(e.currentTarget as HTMLElement).style.transform = 'translateY(0)'
              }}
            >
              <RefreshCw
                size={13}
                style={{
                  animation: retrying ? 'errPanelSpin 0.9s linear infinite' : 'none',
                }}
              />
              {retrying ? 'Retrying…' : 'Retry'}
            </button>
          )}
        </div>
      </div>

      <style>{`
        @keyframes errPanelFadeIn {
          from { opacity: 0; }
          to   { opacity: 1; }
        }
        @keyframes errPanelSlideIn {
          from { opacity: 0; transform: translateY(12px) scale(0.96); }
          to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes errPanelIconPulse {
          0%, 100% { box-shadow: 0 0 0 0 color-mix(in srgb, ${accent} 0%, transparent); }
          50%      { box-shadow: 0 0 0 6px color-mix(in srgb, ${accent} 13%, transparent); }
        }
        @keyframes errPanelSpin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}

// Classify an axios error into an ErrorKind for correct icon/color
export const classifyError = (err: any): { kind: ErrorKind; message: string; details?: string } => {
  if (!err) return { kind: 'unknown', message: 'Unknown error' }

  // Network / connection failure (no response)
  if (err.code === 'ERR_NETWORK' || err.message === 'Network Error' || !err.response) {
    return {
      kind: 'network',
      message: 'Cannot reach the backend. Check your connection or the server.',
      details: err.message,
    }
  }

  const status = err.response?.status
  const detail = err.response?.data?.detail

  // Structured detail (our backend emits objects for DataUnavailable)
  if (detail && typeof detail === 'object') {
    if (detail.error === 'data_unavailable') {
      const src = detail.source === 'mysql' ? 'MySQL' : detail.source
      return {
        kind: detail.source === 'weather' ? 'weather' : 'data',
        message: detail.message || `Data unavailable from ${src}`,
        details: `Source: ${src} | Region: ${detail.region || '—'}`,
      }
    }
    return {
      kind: 'server',
      message: detail.message || 'Server error',
      details: JSON.stringify(detail, null, 2),
    }
  }

  // Plain string detail (FastAPI default)
  const msg = typeof detail === 'string' ? detail : err.message || `HTTP ${status}`

  if (status === 503) return { kind: 'data', message: msg, details: `HTTP ${status}` }
  if (status === 422) return { kind: 'data', message: msg, details: `HTTP ${status}` }
  if (status >= 500)  return { kind: 'server', message: msg, details: `HTTP ${status}` }
  if (/weather/i.test(msg)) return { kind: 'weather', message: msg, details: `HTTP ${status}` }

  return { kind: 'unknown', message: msg, details: `HTTP ${status}` }
}
