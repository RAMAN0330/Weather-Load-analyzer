import React, { useState, useEffect } from 'react';
import { Activity, Eye, EyeOff, Lock, Mail, User, Zap, BarChart3, CloudRain, TrendingUp, Cpu, Shield, ArrowRight } from 'lucide-react';
import { useAuthStore } from './authStore';
import ThemeModeSegmented from '../../components/ThemeModeSegmented';

/* ─── tokens ─────────────────────────────────────────────────── */
/* Theme-aware: every colour is a CSS variable from index.css (Light / One Dark Pro). */
const BRAND = '#F07825'; // Forecast Studio brand-mark tile only (fixed in both themes)
const C = {
  bg:      'var(--bg)',
  panel:   'var(--bg-panel)',
  surf:    'var(--bg-surface)',
  b0:      'var(--outline)',
  b1:      'var(--outline-light)',
  accent:  'var(--accent)',
  accentFg:'var(--accent-fg)',
  warm:    'var(--tone-warm)',
  green:   'var(--success)',
  purple:  'var(--accent2)',
  yellow:  'var(--warning)',
  text:    'var(--text)',
  sub:     'var(--text-muted)',
  dim:     'var(--text-dim)',
  danger:  'var(--danger)',
  sans:    "'Plus Jakarta Sans', sans-serif",
  mono:    "'SF Mono','Fira Code',monospace",
};

/* `${c}22`-style hex alpha breaks with var() — mix instead */
const tint = (c, pct) => `color-mix(in srgb, ${c} ${pct}%, transparent)`;

/* ─── sparkline ──────────────────────────────────────────────── */
function Spark({ color, w = 96, h = 34 }) {
  const pts = [10, 22, 15, 28, 11, 32, 20, 16, 25, 22, 17, 30, 21, 12, 27, 22, 18, 26];
  const mx = Math.max(...pts), mn = Math.min(...pts);
  const ys = pts.map(v => h - 4 - ((v - mn) / (mx - mn)) * (h - 10));
  const xs = pts.map((_, i) => (i / (pts.length - 1)) * w);
  const line = ys.map((y, i) => `${i ? 'L' : 'M'}${xs[i].toFixed(1)} ${y.toFixed(1)}`).join(' ');
  const area = `${line} L${w} ${h} L0 ${h} Z`;
  const id = `sp${color.replace(/[^a-z0-9]/gi, '')}`;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ overflow: 'visible', flexShrink: 0 }}>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" style={{ stopColor: color, stopOpacity: 0.22 }} />
          <stop offset="100%" style={{ stopColor: color, stopOpacity: 0 }} />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${id})`} />
      <path d={line} fill="none" style={{ stroke: color }} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={xs.at(-1)} cy={ys.at(-1)} r="3" style={{ fill: color }} />
    </svg>
  );
}

/* ─── metric tile ────────────────────────────────────────────── */
function Tile({ icon: Icon, label, value, note, color, spark }) {
  return (
    <div style={{
      background: 'rgba(var(--panel-rgb), 0.55)',
      backdropFilter: 'blur(8px)',
      border: `1px solid ${C.b0}`,
      boxShadow: '0 4px 14px rgba(var(--shadow-rgb), 0.06)',
      borderRadius: 16,
      padding: '18px 20px',
      display: 'flex', flexDirection: 'column', gap: 12,
      transition: 'transform .2s ease',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 10,
            background: tint(color, 10), border: `1px solid ${tint(color, 22)}`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Icon size={14} style={{ color }} />
          </div>
          <span style={{ fontSize: 10, color: C.sub, fontWeight: 600, fontFamily: C.mono, letterSpacing: 0.8, textTransform: 'uppercase' }}>
            {label}
          </span>
        </div>
        {spark && <Spark color={color} />}
      </div>
      <div>
        <div style={{ fontSize: 22, fontWeight: 700, color: C.text, letterSpacing: -0.6, lineHeight: 1 }}>{value}</div>
        {note && <div style={{ fontSize: 11, color: C.dim, marginTop: 6, fontWeight: 500 }}>{note}</div>}
      </div>
    </div>
  );
}

/* ─── tag ────────────────────────────────────────────────────── */
function Tag({ children, color }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center',
      fontSize: 10.5, fontWeight: 700, padding: '5px 12px',
      borderRadius: 8, background: tint(color, 9),
      color, border: `1px solid ${tint(color, 24)}`,
      letterSpacing: 0.2, fontFamily: C.mono,
    }}>{children}</span>
  );
}

/* ═════════════════════════════════════════════════════════════
   PAGE
   ═════════════════════════════════════════════════════════════ */
export default function LoginPage({ onAuth }) {
  const [mode, setMode]     = useState('login');
  const [form, setForm]     = useState({ username: '', email: '', password: '' });
  const [showPw, setShowPw] = useState(false);
  const [error, setError]   = useState('');
  const [busy, setBusy]     = useState(false);
  const [foc, setFoc]       = useState('');
  const [tick, setTick]     = useState(0);

  const { login, register } = useAuthStore();

  useEffect(() => {
    const t = setInterval(() => setTick(n => n + 1), 1500);
    return () => clearInterval(t);
  }, []);

  const onChange = e => { setForm(f => ({ ...f, [e.target.name]: e.target.value })); setError(''); };

  const onSubmit = async e => {
    e.preventDefault(); setError(''); setBusy(true);
    try {
      if (mode === 'login') {
        await login(form.username, form.password);
      } else {
        await register(form.username, form.email, form.password);
        await login(form.username, form.password);
      }
      onAuth();
    } catch (err) {
      const r = err?.response?.data?.detail;
      setError(typeof r === 'string' ? r : Array.isArray(r) ? r.map(d => d.msg).join(', ') : 'Authentication failed.');
    } finally { setBusy(false); }
  };

  const bars = Array.from({ length: 22 }, (_, i) => ({
    h: 14 + Math.sin((i + tick) * 0.68) * 13 + Math.random() * 6,
    on: i === 21 - (tick % 5),
  }));

  const inp = name => ({
    width: '100%', boxSizing: 'border-box',
    background: C.surf,
    border: `1.5px solid ${foc === name ? C.accent : C.b1}`,
    borderRadius: 12,
    padding: '12px 14px 12px 42px',
    fontSize: 14,
    fontFamily: C.sans,
    color: C.text,
    outline: 'none',
    transition: 'all .25s cubic-bezier(0.2, 0.8, 0.2, 1)',
    boxShadow: foc === name ? '0 0 0 4px rgba(var(--accent-rgb), 0.2)' : 'none',
    letterSpacing: 0.1,
  });

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        * { box-sizing: border-box; }
        @keyframes fadeIn { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
        @keyframes slideL { from{opacity:0;transform:translateX(-30px)} to{opacity:1;transform:translateX(0)} }
        @keyframes slideR { from{opacity:0;transform:translateX(30px)}  to{opacity:1;transform:translateX(0)} }
        @keyframes blink  { 0%,100%{opacity:1} 50%{opacity:.3} }
        @keyframes spin   { to{transform:rotate(360deg)} }
        .field::placeholder { color:${C.dim}; font-size:13px; opacity: 0.85; }
        .btn-main { transition: all .3s cubic-bezier(0.2, 0.8, 0.2, 1); }
        .btn-main:hover:not(:disabled) { filter:brightness(1.1); transform:translateY(-2px); box-shadow:0 12px 30px rgba(var(--accent-rgb), 0.33) !important; }
        .btn-main:focus-visible, .tab-btn:focus-visible { outline: 2px solid var(--outline-focus); outline-offset: 2px; }
        .btn-main:active:not(:disabled) { transform:translateY(0); }
        .tab-btn { position: relative; z-index: 2; transition: color .3s ease; }
        .anim-item { animation: fadeIn .6s cubic-bezier(0.2, 0.8, 0.2, 1) both; }
        .grid-mask { mask-image: radial-gradient(ellipse 70% 70% at 50% 50%, black 30%, transparent 100%); }
      `}</style>

      <div style={{
        position: 'fixed', inset: 0,
        background: C.bg,
        display: 'flex',
        fontFamily: "'Plus Jakarta Sans', sans-serif",
        color: C.text,
        overflow: 'hidden',
      }}>

        {/* ── backgrounds ── */}
        <div className="grid-mask" style={{
          position: 'absolute', inset: 0, pointerEvents: 'none',
          backgroundImage: 'linear-gradient(rgba(var(--overlay-rgb), 0.05) 1.5px,transparent 1.5px),linear-gradient(90deg,rgba(var(--overlay-rgb), 0.05) 1.5px,transparent 1.5px)',
          backgroundSize: '56px 56px',
        }} />

        <div style={{ position:'absolute', top:'-15%', left:'-10%', width:'50%', height:'50%', borderRadius:'50%', background:'radial-gradient(circle,rgba(var(--accent-rgb), 0.1) 0%,transparent 70%)', pointerEvents:'none', filter:'blur(60px)' }} />
        <div style={{ position:'absolute', bottom:'-15%', right:'-5%', width:'45%', height:'45%', borderRadius:'50%', background:`radial-gradient(circle,${tint(C.purple, 8)} 0%,transparent 70%)`, pointerEvents:'none', filter:'blur(60px)' }} />

        {/* theme mode — labelled and reachable before sign-in */}
        <div style={{ position: 'absolute', top: 20, right: 24, zIndex: 20 }}>
          <ThemeModeSegmented />
        </div>

        {/* ════════════════════════
            LEFT SECTION
        ════════════════════════ */}
        <div style={{
          flex: '0 0 52%',
          display: 'flex', flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '60px 80px',
          borderRight: `1px solid ${C.b0}`,
          position: 'relative', zIndex: 1,
          animation: 'slideL .7s cubic-bezier(0.2, 0.8, 0.2, 1) both',
        }}>

          {/* logo */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <div style={{
              width: 48, height: 48, borderRadius: 14,
              background: BRAND, border: `1px solid ${tint(BRAND, 60)}`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: `0 8px 20px ${tint(BRAND, 30)}`,
            }}>
              <Zap size={22} color="#fff" strokeWidth={2.5} />
            </div>
            <div>
              <div style={{ fontSize: 19, fontWeight: 800, letterSpacing: -0.5, color: C.text }}>Forecast Studio</div>
              <div style={{ fontSize: 11.5, color: C.dim, fontWeight: 600, marginTop: 1, letterSpacing: 0.2 }}>Grid Load Forecasting</div>
            </div>
          </div>

          {/* headline */}
          <div style={{ maxWidth: 520 }}>
            <h1 style={{ fontSize: 52, fontWeight: 800, lineHeight: 1.05, letterSpacing: -2.4, color: C.text, margin: '0 0 24px' }}>
              Precision power for<br />
              <span style={{
                background: `linear-gradient(120deg, ${C.accent} 10%, ${C.purple} 90%)`,
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
              }}>hyper-scale grids.</span>
            </h1>
            <p style={{ fontSize: 15, color: C.sub, lineHeight: 1.7, margin: '0 0 32px', fontWeight: 500 }}>
              Enterprise-grade short-term load forecasting. Harness active T+1 / T+2 
              intelligence with hybrid ML weather-recursive processing.
            </p>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <Tag color={C.warm}>Short-Term Ops</Tag>
              <Tag color={C.accent}>T+2 Analytics</Tag>
              <Tag color={C.green}>Active Monitoring</Tag>
              <Tag color={C.purple}>Hybrid ML</Tag>
            </div>
          </div>

          {/* metrics */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
            {/* Capabilities only: no performance claims on the public login page.
                Measured accuracy belongs behind sign-in, sourced from backtests. */}
            <Tile icon={BarChart3}  label="Resolution"     value="96 blocks"     note="15-minute, every day"         color={C.warm}   />
            <Tile icon={TrendingUp} label="Horizons"       value="T+1 · T+2"     note="Day-ahead and day-after"      color={C.accent} />
            <Tile icon={CloudRain}  label="Weather inputs" value="District-level" note="Load-weighted across Haryana" color={C.green}  />
            <Tile icon={Cpu}        label="Uncertainty"    value="P10–P90"       note="Quantile band for every block" color={C.purple} />
          </div>

          {/* decorative footer — labelled as illustrative, not live data */}
          <div aria-hidden="true">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
              <span style={{ fontSize: 11, color: C.dim, letterSpacing: 1.2, textTransform: 'uppercase', fontWeight: 700, fontFamily: C.mono }}>
                Illustrative 96-block profile
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 3, height: 48 }}>
              {bars.map((b, i) => (
                <div key={i} style={{
                  flex: 1, borderRadius: '4px 4px 0 0',
                  height: b.h,
                  background: b.on
                    ? 'linear-gradient(180deg, var(--accent), rgba(var(--accent-rgb), 0.2))'
                    : 'rgba(var(--overlay-rgb), 0.07)',
                  transition: 'height .4s cubic-bezier(0.2, 0.8, 0.2, 1), background .3s ease',
                }} />
              ))}
            </div>
          </div>
        </div>

        {/* ════════════════════════
            RIGHT SECTION
        ════════════════════════ */}
        <div style={{
          flex: 1,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: '60px',
          position: 'relative', zIndex: 1,
          animation: 'slideR .75s cubic-bezier(0.2, 0.8, 0.2, 1) both',
        }}>
          <div style={{ width: '100%', maxWidth: 410 }}>

            {/* card */}
            <div style={{
              background: 'rgba(var(--panel-rgb), 0.82)',
              backdropFilter: 'blur(16px)',
              border: `1px solid ${C.b0}`,
              borderRadius: 28,
              padding: '44px 40px 40px',
              boxShadow: '0 40px 100px rgba(var(--shadow-rgb), 0.22), inset 0 0 0 1px rgba(var(--overlay-rgb), 0.02)',
              position: 'relative',
              overflow: 'hidden',
            }}>

              {/* glow effect */}
              <div style={{ position:'absolute', top:'-10%', right:'-10%', width:'40%', height:'40%', background:'radial-gradient(circle, rgba(var(--accent-rgb), 0.08) 0%, transparent 70%)', pointerEvents:'none' }} />

              {/* header */}
              <div style={{ marginBottom: 32, position: 'relative' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
                  <div style={{ width: 14, height: 14, borderRadius: 4, background: 'rgba(var(--accent-rgb), 0.14)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Shield size={10} style={{ color: C.accent }} strokeWidth={2.5} />
                  </div>
                  <span style={{ fontSize: 11, color: C.accent, fontWeight: 800, letterSpacing: 1.5, textTransform: 'uppercase', fontFamily: C.mono }}>
                    Identity Verification
                  </span>
                </div>
                <h2 style={{ fontSize: 28, fontWeight: 800, letterSpacing: -0.8, color: C.text, lineHeight: 1.1, marginBottom: 10 }}>
                  {mode === 'login' ? 'Access Terminal' : 'Portal Access'}
                </h2>
                <p style={{ fontSize: 14, color: C.sub, lineHeight: 1.5, fontWeight: 500 }}>
                  {mode === 'login'
                    ? 'Authenticate to enter your dashboard.'
                    : 'Create your operator account to begin.'}
                </p>
              </div>

              {/* tabs (sliding highlight) */}
              <div style={{
                position: 'relative',
                display: 'flex', padding: 4, marginBottom: 32,
                background: C.surf, border: `1px solid ${C.b0}`, borderRadius: 14,
                height: 44,
              }}>
                <div style={{
                  position: 'absolute', top: 4, bottom: 4, left: 4,
                  width: 'calc(50% - 6px)',
                  background: 'linear-gradient(135deg, var(--accent), var(--accent-soft))',
                  borderRadius: 10,
                  transform: `translateX(${mode === 'login' ? '0%' : '100%'})`,
                  transition: 'transform .4s cubic-bezier(0.2, 0.8, 0.2, 1)',
                  boxShadow: '0 4px 15px rgba(var(--accent-rgb), 0.25)',
                  zIndex: 1,
                }} />
                
                {[['login', 'Sign In'], ['register', 'Join Portal']].map(([k, l]) => (
                  <button key={k} className="tab-btn"
                    onClick={() => { setMode(k); setError(''); }}
                    style={{
                      position: 'relative', zIndex: 2,
                      flex: 1, height: '100%', border: 'none', background: 'none',
                      fontSize: 13, fontWeight: 700, letterSpacing: 0.1,
                      cursor: 'pointer', fontFamily: C.sans,
                      color: mode === k ? C.accentFg : C.sub,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}>{l}</button>
                ))}
              </div>

              {/* form */}
              <form onSubmit={onSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

                <div className="anim-item" style={{ animationDelay: '0.1s', display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <label style={{ fontSize: 12, fontWeight: 700, color: C.sub, letterSpacing: 0.3, textTransform: 'uppercase' }}>Username</label>
                  <div style={{ position: 'relative' }}>
                    <User size={16} style={{ position:'absolute', left:14, top:'50%', transform:'translateY(-50%)', color: foc==='username' ? C.accent : C.dim, transition:'color .25s', pointerEvents:'none' }} />
                    <input name="username" value={form.username} onChange={onChange}
                      onFocus={() => setFoc('username')} onBlur={() => setFoc('')}
                      required autoComplete="username" placeholder="Username"
                      className="field" style={inp('username')} />
                  </div>
                </div>

                {mode === 'register' && (
                  <div className="anim-item" style={{ animationDelay: '0.15s', display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <label style={{ fontSize: 12, fontWeight: 700, color: C.sub, letterSpacing: 0.3, textTransform: 'uppercase' }}>Email Address</label>
                    <div style={{ position: 'relative' }}>
                      <Mail size={16} style={{ position:'absolute', left:14, top:'50%', transform:'translateY(-50%)', color: foc==='email' ? C.accent : C.dim, transition:'color .25s', pointerEvents:'none' }} />
                      <input name="email" type="email" value={form.email} onChange={onChange}
                        onFocus={() => setFoc('email')} onBlur={() => setFoc('')}
                        required autoComplete="email" placeholder="you@gnacorp.in"
                        className="field" style={inp('email')} />
                    </div>
                  </div>
                )}

                <div className="anim-item" style={{ animationDelay: '0.2s', display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <label style={{ fontSize: 12, fontWeight: 700, color: C.sub, letterSpacing: 0.3, textTransform: 'uppercase' }}>Password</label>
                  <div style={{ position: 'relative' }}>
                    <Lock size={16} style={{ position:'absolute', left:14, top:'50%', transform:'translateY(-50%)', color: foc==='password' ? C.accent : C.dim, transition:'color .25s', pointerEvents:'none' }} />
                    <input name="password" type={showPw ? 'text' : 'password'}
                      value={form.password} onChange={onChange}
                      onFocus={() => setFoc('password')} onBlur={() => setFoc('')}
                      required minLength={6}
                      autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                      placeholder="••••••••"
                      className="field" style={{ ...inp('password'), paddingRight: 44 }} />
                    <button type="button" onClick={() => setShowPw(v => !v)} style={{
                      position: 'absolute', right: 14, top: '50%', transform: 'translateY(-50%)',
                      background: 'none', border: 'none', cursor: 'pointer',
                      color: showPw ? C.accent : C.dim, padding: 4,
                      display: 'flex', alignItems: 'center', transition: 'all .25s',
                    }}>
                      {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  </div>
                </div>

                {error && (
                  <div className="anim-item" style={{
                    background: 'var(--danger-dim)', border: `1px solid ${tint(C.danger, 20)}`,
                    borderRadius: 12, padding: '12px 16px',
                    fontSize: 13, color: C.danger, lineHeight: 1.5, fontWeight: 500,
                  }}>{error}</div>
                )}

                <button type="submit" disabled={busy} className="btn-main anim-item" style={{
                  animationDelay: '0.25s',
                  width: '100%', padding: '15px 0', borderRadius: 14,
                  background: busy ? 'rgba(var(--overlay-rgb), 0.06)' : 'linear-gradient(135deg, var(--accent) 0%, var(--accent-soft) 100%)',
                  border: 'none', cursor: busy ? 'wait' : 'pointer',
                  color: busy ? C.sub : C.accentFg, fontSize: 14.5, fontWeight: 800,
                  fontFamily: C.sans, letterSpacing: 0.5,
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
                  boxShadow: busy ? 'none' : '0 8px 24px rgba(var(--accent-rgb), 0.27)',
                  marginTop: 8,
                }}>
                  {busy
                    ? <><Activity size={16} style={{ animation: 'spin 1.2s linear infinite' }} /> {mode === 'login' ? 'AUTHENTICATING' : 'CREATING IDENTITY'}</>
                    : <>{mode === 'login' ? 'INITIALIZE SIGN IN' : 'REGISTER OPERATOR'} <ArrowRight size={18} strokeWidth={2.5} /></>
                  }
                </button>
              </form>

              {/* divider/status */}
              <div className="anim-item" style={{ animationDelay: '0.35s', marginTop: 32, display: 'flex', alignItems: 'center', gap: 12 }}>
                <div style={{ flex: 1, height: 1, background: C.b0 }} />
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{
                    width: 7, height: 7, borderRadius: '50%', background: C.green,
                    boxShadow: `0 0 8px ${C.green}`,
                    animation: 'blink 3s ease-in-out infinite',
                  }} />
                  <span style={{ fontSize: 10, color: C.dim, letterSpacing: 1.2, fontWeight: 800, fontFamily: C.mono, textTransform: 'uppercase' }}>
                    Secured Node
                  </span>
                </div>
                <div style={{ flex: 1, height: 1, background: C.b0 }} />
              </div>
            </div>

            <div className="anim-item" style={{ animationDelay: '0.45s', marginTop: 24, textAlign: 'center', fontSize: 12, color: C.dim, fontWeight: 500 }}>
              Forecast Studio v2.4 &middot; GNA Energy Operations &middot; {new Date().getFullYear()}
            </div>
          </div>
        </div>

      </div>
    </>
  );
}
