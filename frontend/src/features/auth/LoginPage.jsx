import React, { useState, useEffect } from 'react';
import { Activity, Eye, EyeOff, Lock, Mail, User, Zap, BarChart3, CloudRain, TrendingUp, Cpu, Shield, ArrowRight } from 'lucide-react';
import { useAuthStore } from './authStore';

/* ─── tokens ─────────────────────────────────────────────────── */
const C = {
  bg:      '#09090E',
  panel:   '#0E0E15',
  card:    '#12121A',
  surf:    '#17171F',
  b0:      '#1E1E28',
  b1:      '#28283A',
  accent:  '#F07825',
  blue:    '#5B9FE4',
  green:   '#3DD68C',
  purple:  '#9B7FE8',
  yellow:  '#F5C147',
  text:    '#EDEEF2',
  sub:     '#7E849A',
  dim:     '#44465A',
  danger:  '#E05555',
  sans:    "'Plus Jakarta Sans', sans-serif",
  mono:    "'SF Mono','Fira Code',monospace",
};

/* ─── sparkline ──────────────────────────────────────────────── */
function Spark({ color, w = 96, h = 34 }) {
  const pts = [10, 22, 15, 28, 11, 32, 20, 16, 25, 22, 17, 30, 21, 12, 27, 22, 18, 26];
  const mx = Math.max(...pts), mn = Math.min(...pts);
  const ys = pts.map(v => h - 4 - ((v - mn) / (mx - mn)) * (h - 10));
  const xs = pts.map((_, i) => (i / (pts.length - 1)) * w);
  const line = ys.map((y, i) => `${i ? 'L' : 'M'}${xs[i].toFixed(1)} ${y.toFixed(1)}`).join(' ');
  const area = `${line} L${w} ${h} L0 ${h} Z`;
  const id = `sp${color.replace('#', '')}`;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ overflow: 'visible', flexShrink: 0 }}>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity=".22" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${id})`} />
      <path d={line} fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={xs.at(-1)} cy={ys.at(-1)} r="3" fill={color} />
    </svg>
  );
}

/* ─── metric tile ────────────────────────────────────────────── */
function Tile({ icon: Icon, label, value, note, color, spark }) {
  return (
    <div style={{
      background: 'rgba(23, 23, 31, 0.4)',
      backdropFilter: 'blur(8px)',
      border: `1px solid rgba(255, 255, 255, 0.04)`,
      borderRadius: 16,
      padding: '18px 20px',
      display: 'flex', flexDirection: 'column', gap: 12,
      transition: 'transform .2s ease',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 32, height: 32, borderRadius: 10,
            background: `${color}18`, border: `1px solid ${color}22`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Icon size={14} color={color} />
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
      borderRadius: 8, background: `${color}15`,
      color, border: `1px solid ${color}25`,
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
    background: 'rgba(255, 255, 255, 0.02)',
    border: `1.5px solid ${foc === name ? C.accent : 'rgba(255, 255, 255, 0.05)'}`,
    borderRadius: 12,
    padding: '12px 14px 12px 42px',
    fontSize: 14,
    fontFamily: C.sans,
    color: C.text,
    outline: 'none',
    transition: 'all .25s cubic-bezier(0.2, 0.8, 0.2, 1)',
    boxShadow: foc === name ? `0 0 0 4px ${C.accent}15` : 'none',
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
        .field::placeholder { color:${C.dim}; font-size:13px; opacity: 0.6; }
        .btn-main { transition: all .3s cubic-bezier(0.2, 0.8, 0.2, 1); }
        .btn-main:hover:not(:disabled) { filter:brightness(1.1); transform:translateY(-2px); box-shadow:0 12px 30px ${C.accent}55 !important; }
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
          backgroundImage: `linear-gradient(${C.b0}77 1.5px,transparent 1.5px),linear-gradient(90deg,${C.b0}77 1.5px,transparent 1.5px)`,
          backgroundSize: '56px 56px',
        }} />

        <div style={{ position:'absolute', top:'-15%', left:'-10%', width:'50%', height:'50%', borderRadius:'50%', background:`radial-gradient(circle,${C.accent}0D 0%,transparent 70%)`, pointerEvents:'none', filter:'blur(60px)' }} />
        <div style={{ position:'absolute', bottom:'-15%', right:'-5%', width:'45%', height:'45%', borderRadius:'50%', background:`radial-gradient(circle,${C.blue}0A 0%,transparent 70%)`, pointerEvents:'none', filter:'blur(60px)' }} />

        {/* ════════════════════════
            LEFT SECTION
        ════════════════════════ */}
        <div style={{
          flex: '0 0 52%',
          display: 'flex', flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '60px 80px',
          borderRight: `1px solid rgba(255, 255, 255, 0.04)`,
          position: 'relative', zIndex: 1,
          animation: 'slideL .7s cubic-bezier(0.2, 0.8, 0.2, 1) both',
        }}>

          {/* logo */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <div style={{
              width: 48, height: 48, borderRadius: 14,
              background: `linear-gradient(135deg, ${C.accent}22, ${C.accent}11)`, border: `1px solid ${C.accent}33`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: `0 8px 16px ${C.accent}0D`,
            }}>
              <Zap size={22} color={C.accent} strokeWidth={2.5} />
            </div>
            <div>
              <div style={{ fontSize: 19, fontWeight: 800, letterSpacing: -0.5, color: '#fff' }}>VidyutPragya</div>
              <div style={{ fontSize: 11.5, color: C.dim, fontWeight: 600, marginTop: 1, letterSpacing: 0.2 }}>Grid Intelligence Hub</div>
            </div>
          </div>

          {/* headline */}
          <div style={{ maxWidth: 520 }}>
            <h1 style={{ fontSize: 52, fontWeight: 800, lineHeight: 1.05, letterSpacing: -2.4, color: '#fff', margin: '0 0 24px' }}>
              Precision power for<br />
              <span style={{
                background: `linear-gradient(120deg, ${C.accent} 10%, #FF9F45 90%)`,
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
              }}>hyper-scale grids.</span>
            </h1>
            <p style={{ fontSize: 15, color: C.sub, lineHeight: 1.7, margin: '0 0 32px', fontWeight: 500 }}>
              Enterprise-grade short-term load forecasting. Harness active T+1 / T+2 
              intelligence with hybrid ML weather-recursive processing.
            </p>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <Tag color={C.accent}>Short-Term Ops</Tag>
              <Tag color={C.blue}>T+2 Analytics</Tag>
              <Tag color={C.green}>Active Monitoring</Tag>
              <Tag color={C.purple}>Hybrid ML</Tag>
            </div>
          </div>

          {/* metrics */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
            <Tile icon={BarChart3}  label="Mean Accuracy"  value="97.8%"    note="Last 24 hours"        color={C.accent}  spark />
            <Tile icon={TrendingUp} label="Horizon Scale"  value="Dual View" note="T+1 / T+2 Seamless"   color={C.blue}    spark />
            <Tile icon={CloudRain}  label="Station Feed"   value="Live"      note="12 High-res nodes"    color={C.green}   />
            <Tile icon={Cpu}        label="Process Time"   value="< 2.4s"    note="Complete 96-block sync" color={C.purple}  />
          </div>

          {/* live simulation footer */}
          <div style={{ opacity: 0.8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
              <span style={{
                width: 7, height: 7, borderRadius: '50%',
                background: C.green, display: 'inline-block',
                boxShadow: `0 0 10px ${C.green}`,
                animation: 'blink 2s ease-in-out infinite',
              }} />
              <span style={{ fontSize: 11, color: C.dim, letterSpacing: 1.2, textTransform: 'uppercase', fontWeight: 700, fontFamily: C.mono }}>
                Live Stream simulation
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 3, height: 48 }}>
              {bars.map((b, i) => (
                <div key={i} style={{
                  flex: 1, borderRadius: '4px 4px 0 0',
                  height: b.h,
                  background: b.on
                    ? `linear-gradient(180deg, ${C.accent}, ${C.accent}33)`
                    : 'rgba(255, 255, 255, 0.05)',
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
              background: 'rgba(18, 18, 26, 0.7)',
              backdropFilter: 'blur(16px)',
              border: '1px solid rgba(255, 255, 255, 0.06)',
              borderRadius: 28,
              padding: '44px 40px 40px',
              boxShadow: '0 40px 100px rgba(0,0,0,0.6), inset 0 0 0 1px rgba(255,255,255,0.02)',
              position: 'relative',
              overflow: 'hidden',
            }}>

              {/* glow effect */}
              <div style={{ position:'absolute', top:'-10%', right:'-10%', width:'40%', height:'40%', background:`radial-gradient(circle, ${C.accent}11 0%, transparent 70%)`, pointerEvents:'none' }} />

              {/* header */}
              <div style={{ marginBottom: 32, position: 'relative' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
                  <div style={{ width: 14, height: 14, borderRadius: 4, background: `${C.accent}22`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Shield size={10} color={C.accent} strokeWidth={2.5} />
                  </div>
                  <span style={{ fontSize: 11, color: C.accent, fontWeight: 800, letterSpacing: 1.5, textTransform: 'uppercase', fontFamily: C.mono }}>
                    Identity Verification
                  </span>
                </div>
                <h2 style={{ fontSize: 28, fontWeight: 800, letterSpacing: -0.8, color: '#fff', lineHeight: 1.1, marginBottom: 10 }}>
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
                background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 14,
                height: 44,
              }}>
                <div style={{
                  position: 'absolute', top: 4, bottom: 4, left: 4,
                  width: 'calc(50% - 6px)',
                  background: `linear-gradient(135deg, ${C.accent}, #D96012)`,
                  borderRadius: 10,
                  transform: `translateX(${mode === 'login' ? '0%' : '100%'})`,
                  transition: 'transform .4s cubic-bezier(0.2, 0.8, 0.2, 1)',
                  boxShadow: `0 4px 15px ${C.accent}40`,
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
                      color: mode === k ? '#fff' : C.sub,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}>{l}</button>
                ))}
              </div>

              {/* form */}
              <form onSubmit={onSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

                <div className="anim-item" style={{ animationDelay: '0.1s', display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <label style={{ fontSize: 12, fontWeight: 700, color: C.sub, letterSpacing: 0.3, textTransform: 'uppercase', opacity: 0.8 }}>Username</label>
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
                    <label style={{ fontSize: 12, fontWeight: 700, color: C.sub, letterSpacing: 0.3, textTransform: 'uppercase', opacity: 0.8 }}>Email Address</label>
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
                  <label style={{ fontSize: 12, fontWeight: 700, color: C.sub, letterSpacing: 0.3, textTransform: 'uppercase', opacity: 0.8 }}>Password</label>
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
                    background: `${C.danger}15`, border: `1px solid ${C.danger}33`,
                    borderRadius: 12, padding: '12px 16px',
                    fontSize: 13, color: '#FF7676', lineHeight: 1.5, fontWeight: 500,
                  }}>{error}</div>
                )}

                <button type="submit" disabled={busy} className="btn-main anim-item" style={{
                  animationDelay: '0.25s',
                  width: '100%', padding: '15px 0', borderRadius: 14,
                  background: busy ? 'rgba(255,255,255,0.05)' : `linear-gradient(135deg, ${C.accent} 0%, #D96012 100%)`,
                  border: 'none', cursor: busy ? 'wait' : 'pointer',
                  color: '#fff', fontSize: 14.5, fontWeight: 800,
                  fontFamily: C.sans, letterSpacing: 0.5,
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
                  boxShadow: busy ? 'none' : `0 8px 24px ${C.accent}44`,
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
                <div style={{ flex: 1, height: 1, background: 'rgba(255,255,255,0.04)' }} />
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
                <div style={{ flex: 1, height: 1, background: 'rgba(255,255,255,0.04)' }} />
              </div>
            </div>

            <div className="anim-item" style={{ animationDelay: '0.45s', marginTop: 24, textAlign: 'center', fontSize: 12, color: C.dim, fontWeight: 500, opacity: 0.7 }}>
              VidyutPragya v2.4 &middot; GNA Energy Operations &middot; {new Date().getFullYear()}
            </div>
          </div>
        </div>

      </div>
    </>
  );
}
