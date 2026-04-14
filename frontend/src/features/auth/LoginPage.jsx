import React, { useState, useEffect, useRef } from 'react';
import { Activity, Eye, EyeOff, Lock, Mail, User, Zap, BarChart3, CloudRain, TrendingUp, Shield, Cpu } from 'lucide-react';
import { useAuthStore } from './authStore';

/* ─── tokens ───────────────────────────────────────────────── */
const T = {
  bg:       '#08070C',
  card:     '#111018',
  surface:  '#18161F',
  border:   '#222029',
  borderHi: '#2E2B38',
  accent:   '#F07825',
  accent2:  '#5B9FE4',
  green:    '#34D399',
  purple:   '#C084FC',
  text:     '#ECEEF3',
  sub:      '#9096AE',
  muted:    '#525666',
  danger:   '#F87171',
  font:     "'IBM Plex Mono','Cascadia Code',monospace",
};

/* ─── animated sparkline SVG ──────────────────────────────── */
function SparkLine({ color = T.accent, h = 40, animated = false }) {
  const pts = [8,28,18,35,12,38,22,20,30,32,18,42,28,15,38,30,22,10,35,28,18];
  const max = Math.max(...pts), min = Math.min(...pts);
  const norm = pts.map(v => h - ((v - min) / (max - min)) * (h - 8) - 4);
  const w = 140;
  const xs = pts.map((_, i) => (i / (pts.length - 1)) * w);
  const d = norm.map((y, i) => `${i === 0 ? 'M' : 'L'}${xs[i].toFixed(1)},${y.toFixed(1)}`).join(' ');
  const fill = `${d} L${w},${h} L0,${h} Z`;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} style={{ overflow: 'visible' }}>
      <defs>
        <linearGradient id={`sg${color.replace('#','')}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.25" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={fill} fill={`url(#sg${color.replace('#','')})`} />
      <path d={d} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
        style={animated ? { strokeDasharray: 400, strokeDashoffset: 0, animation: 'drawLine 2.5s ease forwards' } : {}} />
      <circle cx={xs[xs.length - 1]} cy={norm[norm.length - 1]} r="3" fill={color}
        style={{ filter: `drop-shadow(0 0 4px ${color})` }} />
    </svg>
  );
}

/* ─── glowing metric card ─────────────────────────────────── */
function MetricCard({ label, value, sub, color, icon: Icon, spark }) {
  return (
    <div style={{
      background: `linear-gradient(135deg, rgba(255,255,255,0.025), rgba(255,255,255,0.01))`,
      border: `1px solid ${T.border}`,
      borderRadius: 14, padding: '16px 18px',
      display: 'flex', flexDirection: 'column', gap: 10,
      position: 'relative', overflow: 'hidden',
      transition: 'border-color .2s',
    }}>
      <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 1, background: `linear-gradient(90deg, transparent, ${color}50, transparent)` }} />
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 28, height: 28, borderRadius: 7, background: `${color}18`, border: `1px solid ${color}28`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Icon size={13} color={color} />
          </div>
          <span style={{ fontSize: 9, color: T.muted, letterSpacing: 1.3, textTransform: 'uppercase' }}>{label}</span>
        </div>
        {spark && <SparkLine color={color} h={32} animated />}
      </div>
      <div>
        <div style={{ fontSize: 20, fontWeight: 700, color: T.text, letterSpacing: -0.5, lineHeight: 1 }}>{value}</div>
        {sub && <div style={{ fontSize: 9, color: T.muted, marginTop: 4, letterSpacing: 0.4 }}>{sub}</div>}
      </div>
    </div>
  );
}

/* ─── floating data badge ─────────────────────────────────── */
function Badge({ children, color }) {
  return (
    <span style={{
      fontSize: 9, fontWeight: 700, padding: '3px 9px', borderRadius: 999,
      background: `${color}18`, color, border: `1px solid ${color}30`, letterSpacing: 0.8,
      textTransform: 'uppercase',
    }}>{children}</span>
  );
}

/* ═══════════════════════════════════════════════════════════
   MAIN
═══════════════════════════════════════════════════════════ */
export default function LoginPage({ onAuth }) {
  const [mode, setMode]       = useState('login');
  const [form, setForm]       = useState({ username: '', email: '', password: '' });
  const [showPw, setShowPw]   = useState(false);
  const [error, setError]     = useState('');
  const [loading, setLoading] = useState(false);
  const [focus, setFocus]     = useState('');
  const [tick, setTick]       = useState(0);

  const { login, register } = useAuthStore();

  // live clock tick for the animated "live" bar
  useEffect(() => { const id = setInterval(() => setTick(t => t + 1), 1800); return () => clearInterval(id); }, []);

  const handleChange = e => { setForm(f => ({ ...f, [e.target.name]: e.target.value })); setError(''); };

  const handleSubmit = async e => {
    e.preventDefault(); setError(''); setLoading(true);
    try {
      if (mode === 'login') { await login(form.username, form.password); }
      else { await register(form.username, form.email, form.password); await login(form.username, form.password); }
      onAuth();
    } catch (err) {
      const raw = err?.response?.data?.detail;
      setError(typeof raw === 'string' ? raw : Array.isArray(raw) ? raw.map(d => d.msg).join(', ') : 'Authentication failed.');
    } finally { setLoading(false); }
  };

  const inp = name => ({
    width: '100%', boxSizing: 'border-box',
    background: T.surface,
    border: `1px solid ${focus === name ? T.accent : T.border}`,
    borderRadius: 10, padding: '12px 13px 12px 40px',
    fontSize: 12, fontFamily: T.font, color: T.text, outline: 'none',
    transition: 'border-color .15s, box-shadow .15s', letterSpacing: 0.2,
    boxShadow: focus === name ? `0 0 0 3px rgba(240,120,37,.15)` : 'none',
  });

  const liveBlocks = Array.from({ length: 18 }, (_, i) => ({
    h: 20 + Math.sin((i + tick) * 0.7) * 14 + Math.random() * 8,
    active: i === 17 - (tick % 4),
  }));

  return (
    <>
      <style>{`
        @keyframes fadeUp   { from{opacity:0;transform:translateY(22px)} to{opacity:1;transform:translateY(0)} }
        @keyframes fadeLeft { from{opacity:0;transform:translateX(-18px)} to{opacity:1;transform:translateX(0)} }
        @keyframes fadeRight{ from{opacity:0;transform:translateX(18px)} to{opacity:1;transform:translateX(0)} }
        @keyframes pulse    { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.75)} }
        @keyframes spin     { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
        @keyframes drawLine { from{stroke-dashoffset:400} to{stroke-dashoffset:0} }
        @keyframes scanLine { 0%{top:0%} 100%{top:100%} }
        @keyframes shimmer  { 0%{background-position:-200% center} 100%{background-position:200% center} }
        .vp-inp::placeholder { color:${T.muted}; }
        .vp-inp::-webkit-input-placeholder { color:${T.muted}; }
        .vp-btn:hover:not(:disabled){ filter:brightness(1.12); box-shadow:0 6px 28px rgba(240,120,37,.55)!important; transform:translateY(-1px); }
        .vp-tab:hover{ color:${T.sub}!important; }
      `}</style>

      <div style={{ position:'fixed', inset:0, background:T.bg, display:'flex', fontFamily:T.font, color:T.text, overflow:'hidden' }}>

        {/* grid */}
        <div style={{ position:'absolute', inset:0, pointerEvents:'none',
          backgroundImage:`linear-gradient(to right,rgba(240,120,37,.06) 1px,transparent 1px),linear-gradient(to bottom,rgba(240,120,37,.06) 1px,transparent 1px)`,
          backgroundSize:'52px 52px' }} />

        {/* radial glow top-center */}
        <div style={{ position:'absolute', top:'-15%', left:'30%', width:700, height:700, borderRadius:'50%',
          background:'radial-gradient(circle,rgba(240,120,37,.07) 0%,transparent 60%)', pointerEvents:'none' }} />
        <div style={{ position:'absolute', bottom:'-12%', right:'5%', width:500, height:500, borderRadius:'50%',
          background:'radial-gradient(circle,rgba(91,159,228,.055) 0%,transparent 60%)', pointerEvents:'none' }} />

        {/* scan line */}
        <div style={{ position:'absolute', left:0, right:0, height:1, background:'linear-gradient(90deg,transparent,rgba(240,120,37,.18),transparent)',
          animation:'scanLine 8s linear infinite', pointerEvents:'none', zIndex:0 }} />

        {/* ══════════════════════════════════════
            LEFT — brand + data showcase
        ══════════════════════════════════════ */}
        <div style={{
          flex:'0 0 54%', display:'flex', flexDirection:'column', justifyContent:'center',
          padding:'56px 68px', position:'relative', zIndex:1,
          borderRight:`1px solid ${T.border}`,
          animation:'fadeLeft .6s cubic-bezier(.22,1,.36,1) both',
        }}>

          {/* logo */}
          <div style={{ display:'flex', alignItems:'center', gap:14, marginBottom:52 }}>
            <div style={{
              width:54, height:54, borderRadius:15,
              background:`linear-gradient(145deg,rgba(240,120,37,.22),rgba(240,120,37,.06))`,
              border:`1px solid rgba(240,120,37,.35)`,
              display:'flex', alignItems:'center', justifyContent:'center',
              boxShadow:`0 0 32px rgba(240,120,37,.18), inset 0 1px 0 rgba(255,255,255,.06)`,
            }}>
              <Zap size={24} color={T.accent} />
            </div>
            <div>
              <div style={{ fontSize:24, fontWeight:700, letterSpacing:-1, color:T.text, lineHeight:1 }}>VidyutPragya</div>
            </div>
          </div>

          {/* headline */}
          <div style={{ marginBottom:44 }}>
            <div style={{ fontSize:38, fontWeight:700, lineHeight:1.15, letterSpacing:-1.5, color:T.text, marginBottom:16 }}>
              Smarter grids.<br />
              Sharper forecasts.<br />
              <span style={{
                background:`linear-gradient(90deg,${T.accent},#FFB347)`,
                WebkitBackgroundClip:'text', WebkitTextFillColor:'transparent',
                backgroundClip:'text',
              }}>Zero surprises.</span>
            </div>
            <div style={{ fontSize:12, color:T.sub, lineHeight:1.9, maxWidth:390 }}>
              Real-time short-term load forecasting with T+1 / T+2 horizon,
              weather-driven analytics, and multi-state grid intelligence.
            </div>
          </div>

          {/* badges row */}
          <div style={{ display:'flex', gap:8, flexWrap:'wrap', marginBottom:28 }}>
            <Badge color={T.accent}>96-Block</Badge>
            <Badge color={T.accent2}>T+1 · T+2</Badge>
            <Badge color={T.green}>Live Ops</Badge>
            <Badge color={T.purple}>Weather AI</Badge>
            <Badge color="#FBBF24">Multi-State</Badge>
          </div>

          {/* metric cards */}
          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:10, maxWidth:430 }}>
            <MetricCard icon={BarChart3}  label="Avg MAPE"       value="≤ 2.4%"        sub="30-day rolling"        color={T.accent}  spark />
            <MetricCard icon={TrendingUp} label="Horizon"        value="T+1 / T+2"     sub="Day-ahead & +2"        color={T.accent2} spark />
            <MetricCard icon={CloudRain}  label="Weather blocks" value="96 / day"       sub="15-min intraday"       color={T.green}   />
            <MetricCard icon={Cpu}        label="ML Engine"      value="Ensemble"       sub="Hybrid AI + baseline"  color={T.purple}  />
          </div>

          {/* live mini bar chart */}
          <div style={{ marginTop:32, display:'flex', flexDirection:'column', gap:8 }}>
            <div style={{ fontSize:9, color:T.muted, letterSpacing:1.4, textTransform:'uppercase', display:'flex', alignItems:'center', gap:8 }}>
              <div style={{ width:6, height:6, borderRadius:'50%', background:T.green, boxShadow:`0 0 7px ${T.green}`, animation:'pulse 2s ease-in-out infinite' }} />
              Live load simulation
            </div>
            <div style={{ display:'flex', alignItems:'flex-end', gap:2, height:46 }}>
              {liveBlocks.map((b, i) => (
                <div key={i} style={{
                  flex:1, borderRadius:'3px 3px 0 0',
                  height: b.h,
                  background: b.active
                    ? `linear-gradient(180deg,${T.accent},${T.accent}88)`
                    : `rgba(255,255,255,0.07)`,
                  boxShadow: b.active ? `0 0 8px ${T.accent}66` : 'none',
                  transition:'height .4s ease,background .3s ease',
                }} />
              ))}
            </div>
          </div>
        </div>

        {/* ══════════════════════════════════════
            RIGHT — form panel
        ══════════════════════════════════════ */}
        <div style={{
          flex:1, display:'flex', flexDirection:'column',
          alignItems:'center', justifyContent:'center',
          padding:'48px 56px', position:'relative', zIndex:1,
          animation:'fadeRight .65s cubic-bezier(.22,1,.36,1) both',
        }}>
          <div style={{ width:'100%', maxWidth:368 }}>

            {/* card */}
            <div style={{
              background:`linear-gradient(160deg, rgba(20,19,26,0.97), rgba(14,13,18,0.99))`,
              border:`1px solid ${T.borderHi}`,
              borderRadius:20, padding:'36px 32px',
              boxShadow:`0 40px 80px rgba(0,0,0,.55), 0 0 0 1px rgba(255,255,255,0.025) inset`,
              position:'relative', overflow:'hidden',
            }}>

              {/* top accent line */}
              <div style={{ position:'absolute', top:0, left:0, right:0, height:2,
                background:`linear-gradient(90deg,transparent 0%,${T.accent} 35%,${T.accent2} 65%,transparent 100%)`,
                borderRadius:'20px 20px 0 0' }} />

              {/* corner glows */}
              <div style={{ position:'absolute', top:-40, right:-40, width:130, height:130, borderRadius:'50%',
                background:`radial-gradient(circle,rgba(240,120,37,.1) 0%,transparent 70%)`, pointerEvents:'none' }} />
              <div style={{ position:'absolute', bottom:-40, left:-40, width:110, height:110, borderRadius:'50%',
                background:`radial-gradient(circle,rgba(91,159,228,.08) 0%,transparent 70%)`, pointerEvents:'none' }} />

              {/* header */}
              <div style={{ marginBottom:28, position:'relative' }}>
                <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:6 }}>
                  <Shield size={14} color={T.accent} />
                  <span style={{ fontSize:9, color:T.accent, letterSpacing:1.6, textTransform:'uppercase', fontWeight:700 }}>Secure Access</span>
                </div>
                <div style={{ fontSize:20, fontWeight:700, letterSpacing:-0.6, color:T.text, marginBottom:5 }}>
                  {mode === 'login' ? 'Welcome back' : 'Create account'}
                </div>
                <div style={{ fontSize:11, color:T.muted }}>
                  {mode === 'login' ? 'Enter your credentials to continue' : 'Fill in the details below to register'}
                </div>
              </div>

              {/* tab switcher */}
              <div style={{
                display:'flex', gap:3, padding:4, marginBottom:26,
                background:'rgba(0,0,0,.4)', border:`1px solid ${T.border}`, borderRadius:12,
              }}>
                {[{key:'login',label:'Sign In'},{key:'register',label:'Register'}].map(({key,label}) => (
                  <button key={key} className="vp-tab" onClick={() => { setMode(key); setError(''); }} style={{
                    flex:1, padding:'9px 0', borderRadius:9, fontSize:11, fontWeight:700,
                    letterSpacing:0.4, cursor:'pointer', border:'none', fontFamily:T.font,
                    background: mode===key ? T.accent : 'transparent',
                    color: mode===key ? '#fff' : T.muted,
                    boxShadow: mode===key ? `0 2px 16px rgba(240,120,37,.45)` : 'none',
                    transition:'all .15s',
                  }}>{label}</button>
                ))}
              </div>

              {/* fields */}
              <form onSubmit={handleSubmit} style={{ display:'flex', flexDirection:'column', gap:15 }}>

                {/* username */}
                <div style={{ display:'flex', flexDirection:'column', gap:5 }}>
                  <label style={{ fontSize:9, fontWeight:700, letterSpacing:1.4, textTransform:'uppercase', color:T.muted }}>Username</label>
                  <div style={{ position:'relative' }}>
                    <User size={13} style={{ position:'absolute', left:13, top:'50%', transform:'translateY(-50%)', color: focus==='username' ? T.accent : T.muted, transition:'color .15s', pointerEvents:'none' }} />
                    <input name="username" value={form.username} onChange={handleChange}
                      onFocus={()=>setFocus('username')} onBlur={()=>setFocus('')}
                      required autoComplete="username" placeholder="Enter username"
                      className="vp-inp" style={inp('username')} />
                  </div>
                </div>

                {/* email */}
                {mode === 'register' && (
                  <div style={{ display:'flex', flexDirection:'column', gap:5 }}>
                    <label style={{ fontSize:9, fontWeight:700, letterSpacing:1.4, textTransform:'uppercase', color:T.muted }}>Email</label>
                    <div style={{ position:'relative' }}>
                      <Mail size={13} style={{ position:'absolute', left:13, top:'50%', transform:'translateY(-50%)', color: focus==='email' ? T.accent : T.muted, transition:'color .15s', pointerEvents:'none' }} />
                      <input name="email" type="email" value={form.email} onChange={handleChange}
                        onFocus={()=>setFocus('email')} onBlur={()=>setFocus('')}
                        required autoComplete="email" placeholder="you@gnacorp.in"
                        className="vp-inp" style={inp('email')} />
                    </div>
                  </div>
                )}

                {/* password */}
                <div style={{ display:'flex', flexDirection:'column', gap:5 }}>
                  <label style={{ fontSize:9, fontWeight:700, letterSpacing:1.4, textTransform:'uppercase', color:T.muted }}>Password</label>
                  <div style={{ position:'relative' }}>
                    <Lock size={13} style={{ position:'absolute', left:13, top:'50%', transform:'translateY(-50%)', color: focus==='password' ? T.accent : T.muted, transition:'color .15s', pointerEvents:'none' }} />
                    <input name="password" type={showPw ? 'text' : 'password'} value={form.password} onChange={handleChange}
                      onFocus={()=>setFocus('password')} onBlur={()=>setFocus('')}
                      required minLength={6} autoComplete={mode==='login' ? 'current-password' : 'new-password'}
                      placeholder="Min. 6 characters"
                      className="vp-inp" style={{...inp('password'), paddingRight:40}} />
                    <button type="button" onClick={()=>setShowPw(v=>!v)} style={{
                      position:'absolute', right:11, top:'50%', transform:'translateY(-50%)',
                      background:'none', border:'none', cursor:'pointer',
                      color: showPw ? T.accent : T.muted, padding:2, display:'flex', alignItems:'center', transition:'color .15s',
                    }}>
                      {showPw ? <EyeOff size={13}/> : <Eye size={13}/>}
                    </button>
                  </div>
                </div>

                {/* error */}
                {error && (
                  <div style={{
                    background:'rgba(248,113,113,.08)', border:'1px solid rgba(248,113,113,.22)',
                    borderRadius:9, padding:'9px 13px', fontSize:11, color:T.danger, lineHeight:1.5,
                  }}>{error}</div>
                )}

                {/* divider */}
                <div style={{ height:1, background:`linear-gradient(90deg,transparent,${T.border},transparent)` }} />

                {/* submit */}
                <button type="submit" disabled={loading} className="vp-btn" style={{
                  width:'100%', padding:'13px 0', borderRadius:11,
                  background: loading ? `rgba(240,120,37,.4)` : `linear-gradient(135deg,${T.accent},#E8651A)`,
                  border:'none', cursor: loading ? 'not-allowed' : 'pointer',
                  color:'#fff', fontSize:12, fontWeight:700, fontFamily:T.font, letterSpacing:0.6,
                  display:'flex', alignItems:'center', justifyContent:'center', gap:9,
                  boxShadow: loading ? 'none' : `0 4px 22px rgba(240,120,37,.38)`,
                  transition:'all .15s',
                }}>
                  {loading ? (
                    <><Activity size={13} style={{animation:'spin 1s linear infinite'}} />{mode==='login' ? 'Authenticating…' : 'Creating account…'}</>
                  ) : (
                    <>{mode==='login' ? 'Sign In' : 'Create Account'} <span style={{opacity:.75}}>→</span></>
                  )}
                </button>
              </form>

              {/* footer inside card */}
              <div style={{ marginTop:24, display:'flex', alignItems:'center', justifyContent:'center', gap:10 }}>
                <div style={{ flex:1, height:1, background:T.border }} />
                <div style={{ display:'flex', alignItems:'center', gap:6 }}>
                  <div style={{ width:5, height:5, borderRadius:'50%', background:T.green, boxShadow:`0 0 6px ${T.green}`, animation:'pulse 2.2s ease-in-out infinite' }} />
                  <span style={{ fontSize:9, color:T.muted, letterSpacing:1 }}>SYSTEM ONLINE</span>
                </div>
                <div style={{ flex:1, height:1, background:T.border }} />
              </div>
            </div>

            {/* below card */}
            <div style={{ marginTop:18, textAlign:'center', fontSize:9, color:T.muted, letterSpacing:0.8 }}>
              VidyutPragya · GNA Energy · {new Date().getFullYear()}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
