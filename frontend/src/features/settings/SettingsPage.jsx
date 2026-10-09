import React, { useEffect, useMemo, useState } from 'react';
import { CheckCircle2, Database, Globe, Plus, Shield, UserCircle2, Workflow } from 'lucide-react';
import {
  PageShell as VpPageShell,
  PageHeader as VpPageHeader,
  Pill as VpPill,
} from '../../components/page/PagePrimitives.jsx';

const TABS = [
  { id: 'profile', label: 'User Profile' },
  { id: 'databases', label: 'Data Source' },
  { id: 'workspace', label: 'Workspace' },
];

/* Theme-aware: colours are CSS variables from index.css (Light / One Dark Pro). */
const tint = (c, pct) => `color-mix(in srgb, ${c} ${pct}%, transparent)`;
const PANEL_GRAD = 'linear-gradient(180deg, rgba(var(--panel-rgb), 0.98), rgba(var(--panel-rgb), 0.96))';

const S = {
  page: {
    fontFamily: "'IBM Plex Mono', monospace",
    color: 'var(--text)',
    minHeight: 0,
    height: '100%',
    overflowX: 'hidden',
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
    padding: '8px 0 16px',
  },
  hero: {
    margin: '0 16px',
    background: `radial-gradient(circle at top right, rgba(var(--accent-rgb), 0.12), transparent 34%), radial-gradient(circle at bottom left, ${tint('var(--tone-warm)', 12)}, transparent 34%), ${PANEL_GRAD}`,
    borderRadius: 18,
    border: '1px solid var(--outline)',
    padding: '20px 22px',
    boxShadow: '0 18px 40px rgba(var(--shadow-rgb), 0.12)',
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: 16,
    flexWrap: 'wrap',
  },
  heroTitle: { fontSize: 28, fontWeight: 700, lineHeight: 1.08, color: 'var(--text)' },
  heroSub: { fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.65, marginTop: 8, maxWidth: 780 },
  badge: (color) => ({
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    fontSize: 9,
    fontWeight: 700,
    padding: '5px 10px',
    borderRadius: 999,
    background: tint(color, 10),
    color,
    border: `1px solid ${tint(color, 22)}`,
  }),
  workspace: {
    margin: '0 16px',
    background: PANEL_GRAD,
    borderRadius: 16,
    border: '1px solid var(--outline)',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    boxShadow: '0 18px 40px rgba(var(--shadow-rgb), 0.12)',
  },
  workspaceHeader: {
    padding: '18px 20px 14px',
    borderBottom: '1px solid var(--outline)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    flexWrap: 'wrap',
  },
  workspaceTitle: { fontSize: 10, fontWeight: 700, letterSpacing: 1.4, textTransform: 'uppercase', color: 'var(--text-secondary)' },
  tabBar: {
    display: 'inline-flex',
    gap: 4,
    padding: 5,
    background: 'var(--bg-surface)',
    border: '1px solid var(--outline)',
    borderRadius: 999,
    flexWrap: 'wrap',
  },
  tab: (active) => ({
    padding: '8px 16px',
    fontSize: 10,
    fontWeight: 600,
    letterSpacing: 0.3,
    cursor: 'pointer',
    border: '1px solid transparent',
    borderRadius: 999,
    background: active ? 'rgba(var(--accent-rgb), 0.14)' : 'transparent',
    color: active ? 'var(--accent)' : 'var(--text-secondary)',
    fontFamily: 'inherit',
  }),
  content: {
    padding: '16px',
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
  },
  grid2: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 0.9fr) minmax(0, 1.1fr)',
    gap: 14,
    alignItems: 'start',
  },
  gridAuto: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
    gap: 12,
  },
  card: {
    background: PANEL_GRAD,
    borderRadius: 14,
    border: '1px solid var(--outline)',
    overflow: 'hidden',
  },
  cardHeader: {
    padding: '14px 16px',
    borderBottom: '1px solid var(--outline)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    flexWrap: 'wrap',
  },
  cardTitle: { fontSize: 10, fontWeight: 700, letterSpacing: 1.4, textTransform: 'uppercase', color: 'var(--text-secondary)' },
  cardBody: { padding: 16 },
  fieldGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
    gap: 10,
  },
  fieldBlock: {
    background: 'var(--bg-surface)',
    border: '1px solid var(--outline)',
    borderRadius: 12,
    padding: '12px 14px',
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  fieldLabel: { fontSize: 9, letterSpacing: 1.2, textTransform: 'uppercase', color: 'var(--text-muted)' },
  input: {
    width: '100%',
    borderRadius: 10,
    border: '1px solid var(--outline)',
    background: 'var(--bg-panel)',
    color: 'var(--text)',
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 12,
    padding: '10px 12px',
    outline: 'none',
  },
  helper: { fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.5 },
  profileHero: {
    background: `radial-gradient(circle at top, rgba(var(--accent-rgb), 0.16), transparent 55%), ${PANEL_GRAD}`,
    border: '1px solid var(--outline)',
    borderRadius: 14,
    padding: 18,
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
    gap: 14,
    minHeight: 100,
  },
  profileMeta: { fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.6 },
  sourceStack: { display: 'flex', flexDirection: 'column', gap: 12 },
  sourceCard: {
    background: PANEL_GRAD,
    border: '1px solid var(--outline)',
    borderRadius: 14,
    padding: 14,
    display: 'flex',
    flexDirection: 'column',
    gap: 12,
  },
  sourceHead: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    flexWrap: 'wrap',
  },
  sourceName: { fontSize: 13, fontWeight: 700, color: 'var(--text)' },
  sourceDesc: { fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6 },
  sourceMeta: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, minmax(0, 1fr))',
    gap: 10,
  },
  smallField: {
    background: 'var(--bg-surface)',
    border: '1px solid var(--outline)',
    borderRadius: 12,
    padding: '10px 12px',
  },
  smallLabel: { fontSize: 9, color: 'var(--text-muted)', letterSpacing: 1.1, textTransform: 'uppercase' },
  smallValue: { marginTop: 8, fontSize: 12, color: 'var(--text)', lineHeight: 1.5, wordBreak: 'break-word' },
  actionBtn: {
    border: '1px solid rgba(var(--accent-rgb), 0.35)',
    background: 'rgba(var(--accent-rgb), 0.12)',
    color: 'var(--accent)',
    borderRadius: 999,
    padding: '9px 14px',
    fontSize: 11,
    fontWeight: 700,
    fontFamily: "'IBM Plex Mono', monospace",
    display: 'inline-flex',
    alignItems: 'center',
    gap: 8,
    cursor: 'pointer',
  },
};


export default function SettingsPage({
  settings,
  config,
  effectiveDate,
  selectedRegion,
  availableRegions = [],
  apiBaseUrl,
}) {
  const [activeTab, setActiveTab] = useState('profile');

  const derivedProfile = useMemo(() => ({
    name: settings?.user?.name || settings?.profile?.name || 'Local Operator',
    role: settings?.user?.role || settings?.profile?.role || 'Workspace Admin',
    email: settings?.user?.email || settings?.profile?.email || 'local-session@forecast.ops',
    team: settings?.user?.team || 'Forecast Operations',
    timezone: settings?.user?.timezone || 'Asia/Kolkata',
    region: selectedRegion || 'All Regions',
  }), [settings, selectedRegion]);

  const [profile, setProfile] = useState(derivedProfile);
  useEffect(() => setProfile(derivedProfile), [derivedProfile]);


  const [activeDbSource, setActiveDbSource] = useState(null); // null = file mode, 'mysql' | 'postgres' = DB mode

  const [databases, setDatabases] = useState([
    {
      id: 'mysql',
      label: 'MySQL',
      host: '',
      port: '3306',
      database: '',
      username: '',
      password: '',
      ssl: 'required',
      description: 'Connect to a MySQL database to load historical and live drawal data instead of reading from CSV files.',
    },
    {
      id: 'postgres',
      label: 'PostgreSQL',
      host: '',
      port: '5432',
      database: '',
      username: '',
      password: '',
      ssl: 'require',
      description: 'Connect to a PostgreSQL database to load historical and live drawal data instead of reading from CSV files.',
    },
  ]);

  const handleProfileChange = (key, value) => {
    setProfile((prev) => ({ ...prev, [key]: value }));
  };

  const handleDatabaseChange = (id, key, value) => {
    setDatabases((prev) => prev.map((db) => (
      db.id === id ? { ...db, [key]: value } : db
    )));
  };

  return (
    <VpPageShell className="settings-page">

      <section style={S.workspace}>
        <div style={S.workspaceHeader}>
          <div style={S.workspaceTitle}>Settings Surface</div>
          <div style={S.tabBar}>
            {TABS.map((tab) => (
              <button key={tab.id} type="button" style={S.tab(activeTab === tab.id)} onClick={() => setActiveTab(tab.id)}>
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        <div style={S.content}>
          {activeTab === 'profile' && (
            <div style={S.grid2}>
              <div style={S.profileHero}>
                <div>
                  <UserCircle2 size={44} style={{ color: 'var(--accent)' }} />
                  <div style={{ marginTop: 14, fontSize: 22, fontWeight: 700 }}>{profile.name}</div>
                  <div style={S.profileMeta}>{profile.email}</div>
                  <div style={{ ...S.profileMeta, marginTop: 10 }}>{profile.team}</div>
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <span style={S.badge('var(--success)')}>{profile.role}</span>
                  <span style={S.badge('var(--accent)')}>{profile.timezone}</span>
                </div>
              </div>

              <div style={S.card}>
                <div style={S.cardHeader}>
                  <div style={S.cardTitle}>Profile Details</div>
                  <span style={S.badge('var(--accent)')}><UserCircle2 size={12} /> Editable</span>
                </div>
                <div style={S.cardBody}>
                  <div style={S.fieldGrid}>
                    <div style={S.fieldBlock}>
                      <label style={S.fieldLabel}>Display Name</label>
                      <input style={S.input} value={profile.name} onChange={(e) => handleProfileChange('name', e.target.value)} />
                    </div>
                    <div style={S.fieldBlock}>
                      <label style={S.fieldLabel}>Email</label>
                      <input style={S.input} value={profile.email} onChange={(e) => handleProfileChange('email', e.target.value)} />
                    </div>
                    <div style={S.fieldBlock}>
                      <label style={S.fieldLabel}>Role</label>
                      <input style={S.input} value={profile.role} onChange={(e) => handleProfileChange('role', e.target.value)} />
                    </div>
                    <div style={S.fieldBlock}>
                      <label style={S.fieldLabel}>Team</label>
                      <input style={S.input} value={profile.team} onChange={(e) => handleProfileChange('team', e.target.value)} />
                    </div>
                    <div style={S.fieldBlock}>
                      <label style={S.fieldLabel}>Default Region</label>
                      <input style={S.input} value={profile.region} onChange={(e) => handleProfileChange('region', e.target.value)} />
                    </div>
                    <div style={S.fieldBlock}>
                      <label style={S.fieldLabel}>Timezone</label>
                      <input style={S.input} value={profile.timezone} onChange={(e) => handleProfileChange('timezone', e.target.value)} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'databases' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {/* Active mode banner */}
              <div style={{
                background: activeDbSource
                  ? `linear-gradient(135deg, ${tint('var(--success)', 8)}, ${tint('var(--success)', 4)})`
                  : `linear-gradient(135deg, ${tint('var(--tone-warm)', 8)}, ${tint('var(--tone-warm)', 4)})`,
                border: `1px solid ${activeDbSource ? tint('var(--success)', 20) : tint('var(--tone-warm)', 20)}`,
                borderRadius: 14,
                padding: '14px 18px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: 12,
                flexWrap: 'wrap',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <Database size={16} style={{ color: activeDbSource ? 'var(--success)' : 'var(--tone-warm)' }} />
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text)' }}>
                      {activeDbSource
                        ? `Using ${databases.find(d => d.id === activeDbSource)?.label} as data source`
                        : 'Using file-based data (CSV)'}
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                      {activeDbSource
                        ? 'Data will be read from the connected database instead of local files.'
                        : 'Configure a database below and click "Use as Data Source" to switch.'}
                    </div>
                  </div>
                </div>
                {activeDbSource && (
                  <button type="button" style={{ ...S.actionBtn, borderColor: tint('var(--text-muted)', 40), background: tint('var(--text-muted)', 10), color: 'var(--text-secondary)' }}
                    onClick={() => setActiveDbSource(null)}>
                    Switch back to File
                  </button>
                )}
              </div>

              {/* DB cards */}
              <div style={S.gridAuto}>
                {databases.map((db) => {
                  const isActive = activeDbSource === db.id;
                  return (
                    <div key={db.id} style={{ ...S.card, border: isActive ? `1px solid ${tint('var(--success)', 40)}` : '1px solid var(--outline)' }}>
                      <div style={S.cardHeader}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                          <Database size={14} style={{ color: isActive ? 'var(--success)' : 'var(--accent)' }} />
                          <div style={S.cardTitle}>{db.label}</div>
                        </div>
                        {isActive && <span style={S.badge('var(--success)')}><CheckCircle2 size={11} /> Active Source</span>}
                      </div>
                      <div style={S.cardBody}>
                        <div style={{ ...S.helper, marginBottom: 14 }}>{db.description}</div>
                        <div style={S.fieldGrid}>
                          <div style={S.fieldBlock}>
                            <label style={S.fieldLabel}>Host</label>
                            <input style={S.input} value={db.host} placeholder="localhost" onChange={(e) => handleDatabaseChange(db.id, 'host', e.target.value)} />
                          </div>
                          <div style={S.fieldBlock}>
                            <label style={S.fieldLabel}>Port</label>
                            <input style={S.input} value={db.port} onChange={(e) => handleDatabaseChange(db.id, 'port', e.target.value)} />
                          </div>
                          <div style={S.fieldBlock}>
                            <label style={S.fieldLabel}>Database Name</label>
                            <input style={S.input} value={db.database} placeholder="forecast_db" onChange={(e) => handleDatabaseChange(db.id, 'database', e.target.value)} />
                          </div>
                          <div style={S.fieldBlock}>
                            <label style={S.fieldLabel}>SSL Mode</label>
                            <input style={S.input} value={db.ssl} onChange={(e) => handleDatabaseChange(db.id, 'ssl', e.target.value)} />
                          </div>
                          <div style={S.fieldBlock}>
                            <label style={S.fieldLabel}>Username</label>
                            <input style={S.input} value={db.username} placeholder="db_user" onChange={(e) => handleDatabaseChange(db.id, 'username', e.target.value)} />
                          </div>
                          <div style={S.fieldBlock}>
                            <label style={S.fieldLabel}>Password</label>
                            <input style={S.input} type="password" value={db.password} placeholder="••••••••" onChange={(e) => handleDatabaseChange(db.id, 'password', e.target.value)} />
                          </div>
                        </div>
                        <div style={{ marginTop: 16, display: 'flex', gap: 10 }}>
                          <button
                            type="button"
                            style={{
                              ...S.actionBtn,
                              ...(isActive
                                ? { borderColor: tint('var(--success)', 27), background: 'var(--success-dim)', color: 'var(--success)' }
                                : {}),
                            }}
                            onClick={() => setActiveDbSource(isActive ? null : db.id)}
                          >
                            {isActive ? <CheckCircle2 size={14} /> : <Database size={14} />}
                            {isActive ? 'Active — Click to Deactivate' : 'Use as Data Source'}
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {activeTab === 'workspace' && (
            <div style={S.gridAuto}>
              <div style={S.card}>
                <div style={S.cardHeader}>
                  <div style={S.cardTitle}>General Defaults</div>
                  <span style={S.badge('var(--tone-warm)')}><Workflow size={12} /> Workspace</span>
                </div>
                <div style={S.cardBody}>
                  <div style={S.fieldGrid}>
                    <div style={S.fieldBlock}>
                      <label style={S.fieldLabel}>Default Effective Date</label>
                      <input style={S.input} value={config?.default_date || effectiveDate || '--'} readOnly />
                    </div>
                    <div style={S.fieldBlock}>
                      <label style={S.fieldLabel}>Latest Full Day</label>
                      <input style={S.input} value={config?.latest_date || '--'} readOnly />
                    </div>
                    <div style={S.fieldBlock}>
                      <label style={S.fieldLabel}>Partial Latest Feed</label>
                      <input style={S.input} value={config?.partial_latest_date || 'Not available'} readOnly />
                    </div>
                    <div style={S.fieldBlock}>
                      <label style={S.fieldLabel}>Available Regions</label>
                      <input style={S.input} value={`${availableRegions.length || 0} regions`} readOnly />
                    </div>
                  </div>
                </div>
              </div>

              <div style={S.card}>
                <div style={S.cardHeader}>
                  <div style={S.cardTitle}>Runtime Links</div>
                  <span style={S.badge('var(--accent)')}><Globe size={12} /> API</span>
                </div>
                <div style={S.cardBody}>
                  <div style={S.fieldGrid}>
                    <div style={{ ...S.fieldBlock, gridColumn: '1 / -1' }}>
                      <label style={S.fieldLabel}>API Base</label>
                      <input style={S.input} value={apiBaseUrl || '--'} readOnly />
                    </div>
                    <div style={{ ...S.fieldBlock, gridColumn: '1 / -1' }}>
                      <label style={S.fieldLabel}>Settings Endpoint</label>
                      <input style={S.input} value={apiBaseUrl ? `${apiBaseUrl}/v2/settings` : '--'} readOnly />
                    </div>
                  </div>
                </div>
              </div>

              <div style={S.card}>
                <div style={S.cardHeader}>
                  <div style={S.cardTitle}>Region Calibration</div>
                  <span style={S.badge('var(--success)')}><Shield size={12} /> Operational</span>
                </div>
                <div style={S.cardBody}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.6 }}>
                    These values feed Capacity Utilisation KPI, Ramp Risk alerts, and Reserve Margin calculations. Defaults shown — update per region commissioning report.
                  </div>
                  <div style={S.fieldGrid}>
                    {[
                      { region: 'Odisha', capacity: 5500, rampLimit: 180, eveningPeak: '68–83' },
                      { region: 'Rajasthan', capacity: 7200, rampLimit: 160, eveningPeak: '70–88 (summer)' },
                      { region: 'Haryana', capacity: 4800, rampLimit: 140, eveningPeak: '66–82' },
                    ].map(({ region, capacity, rampLimit, eveningPeak }) => (
                      <div key={region} style={{ ...S.fieldBlock, background: 'var(--bg)', borderRadius: 10, padding: '12px 14px', border: '1px solid var(--outline)' }}>
                        <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--tone-warm)', marginBottom: 8, letterSpacing: 0.5 }}>{region}</div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10 }}>
                            <span style={{ color: 'var(--text-muted)' }}>Installed Capacity</span>
                            <span style={{ color: 'var(--text)', fontWeight: 600 }}>{capacity.toLocaleString('en-IN')} MW</span>
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10 }}>
                            <span style={{ color: 'var(--text-muted)' }}>Ramp Rate Limit</span>
                            <span style={{ color: selectedRegion?.toLowerCase() === region.toLowerCase() && rampLimit < 160 ? 'var(--danger)' : 'var(--text)', fontWeight: 600 }}>{rampLimit} MW/15min</span>
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10 }}>
                            <span style={{ color: 'var(--text-muted)' }}>Evening Peak Blocks</span>
                            <span style={{ color: 'var(--text)', fontWeight: 600 }}>{eveningPeak}</span>
                          </div>
                          {selectedRegion?.toLowerCase() === region.toLowerCase() && (
                            <div style={{ marginTop: 4, fontSize: 9, color: 'var(--success)', fontWeight: 600 }}>▶ ACTIVE REGION</div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </section>
    </VpPageShell>
  );
}
