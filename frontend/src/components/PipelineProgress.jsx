import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { useAuthStore } from '../features/auth/authStore';

const joinApi = (base, path) =>
  `${String(base || '').replace(/\/$/, '')}${path.startsWith('/') ? path : `/${path}`}`;

const toWebSocketUrl = (base, path) => {
  const target = joinApi(base || '/ml-api', path);
  if (/^https?:\/\//i.test(target)) {
    return target.replace(/^http/i, 'ws');
  }
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${protocol}://${window.location.host}${target}`;
};

const STEPS = [
  { key: 1, label: 'Similarity profile' },
  { key: 2, label: 'Baseline window' },
  { key: 3, label: 'Similar days' },
  { key: 4, label: 'ML baseline (XGBoost/Ridge)' },
  { key: 5, label: 'Weather model' },
  { key: 6, label: 'India adjustments' },
  { key: 7, label: 'Final forecast' },
];

export default function PipelineProgress({ jobId, onResult, apiBase = '/ml-api' }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [lastMessage, setLastMessage] = useState('Connecting...');
  const [status, setStatus] = useState('connecting'); // connecting|running|done|error
  const [elapsed, setElapsed] = useState('0.0');
  const [minimized, setMinimized] = useState(false);
  const wsRef = useRef(null);
  const timerRef = useRef(null);
  const startRef = useRef(Date.now());

  useEffect(() => {
    if (!jobId) return;
    startRef.current = Date.now();
    setCurrentStep(0);
    setLastMessage('Connecting...');
    setStatus('connecting');
    setElapsed('0.0');
    setMinimized(false);

    // Browsers can't set headers on a WebSocket, so the session token rides in the query.
    const token = useAuthStore.getState().token;
    const wsPath = `/v2/ws/forecast/${jobId}${token ? `?token=${encodeURIComponent(token)}` : ''}`;
    const wsUrl = toWebSocketUrl(apiBase, wsPath);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    timerRef.current = setInterval(() => {
      setElapsed(((Date.now() - startRef.current) / 1000).toFixed(1));
    }, 200);

    ws.onopen = () => setStatus('running');

    ws.onmessage = (e) => {
      let msg;
      try {
        msg = JSON.parse(e.data);
      } catch {
        return;
      }

      if (msg.type === 'heartbeat') return;

      if (msg.type === 'progress') {
        setLastMessage(msg.message || '');
        const text = msg.message || '';
        if (text[0] && !isNaN(parseInt(text[0]))) {
          const n = parseInt(text.split('/')[0]);
          if (n >= 1 && n <= 7) setCurrentStep(n);
        }
      } else if (msg.type === 'done' || msg.type === 'cached') {
        clearInterval(timerRef.current);
        setStatus('done');
        setCurrentStep(8); // 8 > 7 so all steps show green
        setLastMessage(msg.message || '✓ Done');
        // Fetch result then notify parent
        axios
          .get(joinApi(apiBase, `/v2/forecast/job/${jobId}/result`), { timeout: 15000 })
          .then((r) => onResult?.(r.data))
          .catch(() => onResult?.(null));
        setTimeout(() => setMinimized(true), 2500);
      } else if (msg.type === 'error') {
        clearInterval(timerRef.current);
        setStatus('error');
        setLastMessage(msg.message || 'Pipeline failed');
        onResult?.(null);
      }
    };

    ws.onerror = () => {
      setStatus('error');
      setLastMessage('WebSocket connection failed');
    };

    return () => {
      ws.onopen = null;
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      } else if (ws.readyState === WebSocket.CONNECTING) {
        // Defer close until handshake completes — closing during CONNECTING causes a browser warning
        ws.addEventListener('open', () => ws.close(), { once: true });
      }
      clearInterval(timerRef.current);
    };
  }, [jobId, apiBase]);

  if (!jobId) return null;

  if (minimized)
    return (
      <button
        onClick={() => setMinimized(false)}
        style={{
          position: 'fixed',
          bottom: 20,
          left: 64,
          zIndex: 9999,
          background: 'var(--bg-elevated)',
          border: '1px solid var(--success)',
          borderRadius: 20,
          padding: '6px 14px',
          color: 'var(--success)',
          fontSize: 13,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          fontFamily: "'IBM Plex Mono', monospace",
          boxShadow: '0 4px 12px rgba(var(--shadow-rgb), 0.4)',
        }}
      >
        ✓ Forecast ready · {elapsed}s
      </button>
    );

  const statusColor = {
    connecting: 'var(--text-muted)',
    running: 'var(--warning)',
    done: 'var(--success)',
    error: 'var(--danger)',
  }[status];
  const statusLabel = {
    connecting: 'Connecting',
    running: `${elapsed}s`,
    done: `Done · ${elapsed}s`,
    error: 'Failed',
  }[status];

  return (
    <div
      style={{
        position: 'fixed',
        bottom: 20,
        left: 64,
        zIndex: 9999,
        width: 300,
        background: 'var(--bg-elevated)',
        border: `1px solid ${status === 'error' ? 'color-mix(in srgb, var(--danger) 50%, transparent)' : 'var(--outline)'}`,
        borderRadius: 12,
        boxShadow: '0 8px 32px rgba(var(--shadow-rgb), 0.6)',
        fontFamily: "'IBM Plex Mono', monospace",
        fontSize: 13,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '9px 12px',
          background: 'var(--bg-surface)',
          borderBottom: '1px solid var(--outline)',
          borderRadius: '12px 12px 0 0',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: statusColor,
              display: 'inline-block',
              animation: status === 'running' ? 'mlpulse 1.2s ease-in-out infinite' : 'none',
            }}
          />
          <span style={{ color: 'var(--text)', fontWeight: 600, fontSize: 13 }}>ML Pipeline</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: statusColor, fontSize: 12 }}>{statusLabel}</span>
          <button
            onClick={() => setMinimized(true)}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              padding: 0,
              fontSize: 14,
              lineHeight: 1,
            }}
          >
            ─
          </button>
        </div>
      </div>

      {/* Progress bar */}
      <div style={{ padding: '8px 12px 4px', display: 'flex', gap: 3 }}>
        {STEPS.map((s) => (
          <div
            key={s.key}
            style={{
              flex: 1,
              height: 3,
              borderRadius: 2,
              background:
                currentStep > s.key ? 'var(--success)' : currentStep === s.key ? 'var(--warning)' : 'var(--outline)',
              transition: 'background 0.4s',
            }}
          />
        ))}
      </div>

      {/* Current message */}
      <div
        style={{
          padding: '3px 12px 8px',
          color: 'var(--text-muted)',
          fontSize: 12,
          minHeight: 18,
          letterSpacing: '0.01em',
        }}
      >
        {lastMessage.replace(/^\d+\/7\s*/, '').replace(/^[▶✓]\s*/, '')}
      </div>

      {/* Steps */}
      <div style={{ padding: '0 12px 12px', display: 'flex', flexDirection: 'column', gap: 6 }}>
        {STEPS.map((s) => {
          const done = currentStep > s.key;
          const active = currentStep === s.key && status === 'running';
          return (
            <div key={s.key} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div
                style={{
                  width: 18,
                  height: 18,
                  borderRadius: '50%',
                  flexShrink: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 10,
                  fontWeight: 700,
                  background: done ? 'var(--success)' : active ? 'color-mix(in srgb, var(--warning) 15%, transparent)' : 'transparent',
                  border: `1px solid ${done ? 'var(--success)' : active ? 'var(--warning)' : 'var(--outline-light)'}`,
                  color: done ? 'var(--accent-fg)' : active ? 'var(--warning)' : 'var(--text-dim)',
                  transition: 'all 0.3s',
                }}
              >
                {done ? '✓' : s.key}
              </div>
              <span
                style={{
                  color: done ? 'var(--success)' : active ? 'var(--text)' : 'var(--text-dim)',
                  fontSize: 12,
                  transition: 'color 0.3s',
                  flex: 1,
                }}
              >
                {s.label}
              </span>
              {active && <span style={{ color: 'var(--warning)', fontSize: 10 }}>●</span>}
            </div>
          );
        })}
      </div>

      <style>{`@keyframes mlpulse{0%,100%{opacity:1}50%{opacity:0.25}}`}</style>
    </div>
  );
}
