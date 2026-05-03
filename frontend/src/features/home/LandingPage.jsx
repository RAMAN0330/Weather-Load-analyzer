import React, { useState } from 'react';
import { Zap, TrendingUp, BarChart3, ArrowRight, Brain, Activity, Database, Shield, Clock, MapPin } from 'lucide-react';
import { useAuthStore } from '../auth/authStore';

const ORANGE = '#F07825';
const BLUE = '#5B9FE4';
const GREEN = '#34D399';

export default function LandingPage({ onSelect }) {
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);
  const [choosing, setChoosing] = useState(false);
  const [hovered, setHovered] = useState(null);

  return (
    <div style={{ minHeight: '100vh', width: '100%', display: 'flex', flexDirection: 'column', background: '#09080E', color: '#F0F2F8', fontFamily: 'inherit', overflow: 'hidden', position: 'relative' }}>

      {/* Ambient background glows */}
      <div style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0 }}>
        <div style={{ position: 'absolute', top: '-10%', left: '30%', width: 700, height: 700, borderRadius: '50%', background: 'radial-gradient(circle, rgba(240,120,37,0.08) 0%, transparent 70%)', filter: 'blur(40px)' }} />
        <div style={{ position: 'absolute', bottom: '-5%', right: '20%', width: 500, height: 500, borderRadius: '50%', background: 'radial-gradient(circle, rgba(91,159,228,0.06) 0%, transparent 70%)', filter: 'blur(40px)' }} />
        {/* Subtle grid */}
        <div style={{
          position: 'absolute', inset: 0,
          backgroundImage: 'linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px)',
          backgroundSize: '60px 60px',
        }} />
      </div>

      {/* Header */}
      <header style={{ position: 'relative', zIndex: 10, borderBottom: '1px solid rgba(255,255,255,0.06)', backdropFilter: 'blur(12px)', background: 'rgba(9,8,14,0.7)' }}>
        <div style={{ maxWidth: 1100, margin: '0 auto', padding: '14px 32px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ width: 34, height: 34, borderRadius: 10, background: ORANGE, display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 20px rgba(240,120,37,0.4)' }}>
              <Zap size={16} color="#fff" />
            </div>
            <span style={{ fontWeight: 700, fontSize: 15, letterSpacing: '-0.01em' }}>VidyutPragya</span>
            <span style={{ fontSize: 10, fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase', color: 'rgba(255,255,255,0.25)', marginLeft: 4, paddingLeft: 10, borderLeft: '1px solid rgba(255,255,255,0.1)' }}>
              Forecast OS
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: GREEN, boxShadow: `0 0 8px ${GREEN}` }} />
              <span style={{ fontSize: 12, color: 'rgba(255,255,255,0.35)' }}>Live</span>
            </div>
            <span style={{ fontSize: 13, color: 'rgba(255,255,255,0.35)', padding: '4px 12px', background: 'rgba(255,255,255,0.05)', borderRadius: 20, border: '1px solid rgba(255,255,255,0.08)' }}>
              {user?.username || 'Guest'}
            </span>
            <button
              onClick={() => logout?.()}
              style={{ fontSize: 12, color: 'rgba(255,255,255,0.3)', background: 'none', border: 'none', cursor: 'pointer', padding: 0, transition: 'color 0.2s' }}
              onMouseEnter={(e) => (e.target.style.color = 'rgba(255,255,255,0.65)')}
              onMouseLeave={(e) => (e.target.style.color = 'rgba(255,255,255,0.3)')}
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      {/* Main */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '60px 32px 80px', position: 'relative', zIndex: 1 }}>

        {/* Hero */}
        <div style={{ textAlign: 'center', maxWidth: 720, width: '100%' }}>

          {/* Badge */}
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 7, padding: '5px 14px', borderRadius: 100, background: 'rgba(240,120,37,0.1)', border: '1px solid rgba(240,120,37,0.22)', marginBottom: 32 }}>
            <Zap size={10} color={ORANGE} />
            <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', color: ORANGE }}>Energy Intelligence Platform</span>
          </div>

          {/* Headline */}
          <h1 style={{ fontSize: 'clamp(40px, 6vw, 64px)', fontWeight: 800, lineHeight: 1.1, letterSpacing: '-0.03em', margin: '0 0 20px' }}>
            AI-Powered Grid
            <br />
            <span style={{ background: `linear-gradient(105deg, ${ORANGE} 0%, #F59E0B 60%, #FBBF24 100%)`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              Load Intelligence
            </span>
          </h1>

          <p style={{ fontSize: 17, lineHeight: 1.7, color: 'rgba(255,255,255,0.48)', maxWidth: 560, margin: '0 auto 40px' }}>
            Sub-1% MAPE day-ahead forecasts with deep analytics for power procurement,
            balancing, and grid operations across Indian states.
          </p>

          {/* Stats row */}
          <div style={{ display: 'flex', justifyContent: 'center', gap: 0, marginBottom: 44 }}>
            {[
              { value: '<1%', label: 'MAPE' },
              { value: '96', label: 'Blocks/day' },
              { value: '28', label: 'States' },
              { value: '15-min', label: 'Resolution' },
            ].map((s, i, arr) => (
              <div key={s.label} style={{ padding: '10px 28px', borderRight: i < arr.length - 1 ? '1px solid rgba(255,255,255,0.08)' : 'none', textAlign: 'center' }}>
                <div style={{ fontSize: 22, fontWeight: 700, color: '#F0F2F8', letterSpacing: '-0.02em' }}>{s.value}</div>
                <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.3)', fontWeight: 500, marginTop: 2, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{s.label}</div>
              </div>
            ))}
          </div>

          {/* CTA */}
          {!choosing ? (
            <button
              onClick={() => setChoosing(true)}
              style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '13px 30px', borderRadius: 12, background: ORANGE, color: '#fff', fontSize: 14, fontWeight: 700, border: 'none', cursor: 'pointer', boxShadow: '0 0 40px rgba(240,120,37,0.35)', transition: 'all 0.2s', letterSpacing: '0.01em' }}
              onMouseEnter={(e) => { e.currentTarget.style.background = '#E06918'; e.currentTarget.style.boxShadow = '0 0 56px rgba(240,120,37,0.5)'; e.currentTarget.style.transform = 'translateY(-1px)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = ORANGE; e.currentTarget.style.boxShadow = '0 0 40px rgba(240,120,37,0.35)'; e.currentTarget.style.transform = 'translateY(0)'; }}
            >
              Get Started <ArrowRight size={15} />
            </button>
          ) : (
            <div style={{ display: 'flex', gap: 16, justifyContent: 'center', flexWrap: 'wrap' }}>
              <ModeCard
                icon={<TrendingUp size={22} color={ORANGE} />}
                iconBg="rgba(240,120,37,0.12)"
                iconBorder="rgba(240,120,37,0.25)"
                hoverBorder="rgba(240,120,37,0.5)"
                accentColor={ORANGE}
                title="Forecast"
                desc="Day-ahead load forecast, scenario simulator & backtest engine"
                tag="Primary"
                onClick={() => onSelect('forecast')}
              />
              <ModeCard
                icon={<BarChart3 size={22} color={BLUE} />}
                iconBg="rgba(91,159,228,0.12)"
                iconBorder="rgba(91,159,228,0.25)"
                hoverBorder="rgba(91,159,228,0.5)"
                accentColor={BLUE}
                title="Analytics"
                desc="Weather, load, combined & similar-day deep dashboards"
                tag="Explore"
                onClick={() => onSelect('analytics')}
              />
            </div>
          )}

          {choosing && (
            <p style={{ marginTop: 20, fontSize: 11, color: 'rgba(255,255,255,0.2)', letterSpacing: '0.02em' }}>
              Your choice is remembered for this session
            </p>
          )}
        </div>

        {/* Info cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, maxWidth: 820, width: '100%', marginTop: 72 }}>
          {[
            {
              Icon: Brain,
              title: 'What',
              color: ORANGE,
              bg: 'rgba(240,120,37,0.08)',
              border: 'rgba(240,120,37,0.18)',
              desc: '15-minute resolution ML pipeline for day-ahead & intra-day load forecasting. 96 blocks/day with weather, calendar, and historical demand signals.',
            },
            {
              Icon: Activity,
              title: 'Why',
              color: GREEN,
              bg: 'rgba(52,211,153,0.08)',
              border: 'rgba(52,211,153,0.18)',
              desc: 'Accurate demand forecasts reduce over-scheduling penalties, optimise bilateral contract volumes, and improve deviation settlement under CERC DSM.',
            },
            {
              Icon: Database,
              title: 'How',
              color: BLUE,
              bg: 'rgba(91,159,228,0.08)',
              border: 'rgba(91,159,228,0.18)',
              desc: 'XGBoost + LightGBM ensemble with temperature sensitivity correction, momentum recalibration, and distilled neural inference — MySQL + SQLite hybrid.',
            },
          ].map(({ Icon, title, color, bg, border, desc }) => (
            <div key={title}
              style={{ padding: '22px', background: 'rgba(255,255,255,0.028)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 16, transition: 'border-color 0.2s, background 0.2s' }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = border; e.currentTarget.style.background = 'rgba(255,255,255,0.045)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.07)'; e.currentTarget.style.background = 'rgba(255,255,255,0.028)'; }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                <div style={{ width: 32, height: 32, borderRadius: 9, background: bg, border: `1px solid ${border}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Icon size={15} color={color} />
                </div>
                <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'rgba(255,255,255,0.6)' }}>{title}</span>
              </div>
              <p style={{ fontSize: 13, lineHeight: 1.65, color: 'rgba(255,255,255,0.4)', margin: 0 }}>{desc}</p>
            </div>
          ))}
        </div>

        {/* Trust bar */}
        <div style={{ display: 'flex', gap: 28, marginTop: 52, flexWrap: 'wrap', justifyContent: 'center' }}>
          {[
            { Icon: Shield, text: 'CERC DSM Compliant' },
            { Icon: Clock, text: 'Real-time Updates' },
            { Icon: MapPin, text: 'Pan-India Coverage' },
          ].map(({ Icon, text }) => (
            <div key={text} style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <Icon size={13} color="rgba(255,255,255,0.2)" />
              <span style={{ fontSize: 12, color: 'rgba(255,255,255,0.25)', fontWeight: 500 }}>{text}</span>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}

function ModeCard({ icon, iconBg, iconBorder, hoverBorder, accentColor, title, desc, tag, onClick }) {
  const [hov, setHov] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 14, padding: '24px 26px',
        background: hov ? 'rgba(255,255,255,0.055)' : 'rgba(255,255,255,0.035)',
        border: `1px solid ${hov ? hoverBorder : 'rgba(255,255,255,0.09)'}`,
        borderRadius: 18, cursor: 'pointer', minWidth: 230, textAlign: 'left',
        transition: 'all 0.22s', transform: hov ? 'translateY(-2px)' : 'translateY(0)',
        boxShadow: hov ? `0 12px 40px rgba(0,0,0,0.3)` : 'none',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
        <div style={{ width: 48, height: 48, borderRadius: 14, background: iconBg, border: `1px solid ${iconBorder}`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {icon}
        </div>
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: accentColor, padding: '3px 9px', background: `${accentColor}18`, borderRadius: 100, border: `1px solid ${accentColor}30` }}>
          {tag}
        </span>
      </div>
      <div>
        <div style={{ fontWeight: 700, fontSize: 16, color: '#F0F2F8', marginBottom: 6 }}>{title}</div>
        <div style={{ fontSize: 13, lineHeight: 1.55, color: 'rgba(255,255,255,0.42)', maxWidth: 200 }}>{desc}</div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginTop: 4 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: accentColor }}>Open</span>
        <ArrowRight size={13} color={accentColor} />
      </div>
    </button>
  );
}
