import React, { useEffect, useMemo, useState } from 'react';
import { Database, Globe, Plus, Shield, UserCircle2, Workflow } from 'lucide-react';

const TABS = [
  { id: 'profile', label: 'User Profile' },
  { id: 'sources', label: 'Data Sources' },
  { id: 'databases', label: 'Databases' },
  { id: 'workspace', label: 'Workspace' },
];

const S = {
  page: {
    fontFamily: "'IBM Plex Mono', monospace",
    color: '#ECEEF3',
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
    background: 'radial-gradient(circle at top right, rgba(91, 159, 228, 0.12), transparent 34%), radial-gradient(circle at bottom left, rgba(240, 120, 37, 0.12), transparent 34%), linear-gradient(180deg, rgba(29, 28, 34, 0.98), rgba(19, 19, 24, 0.98))',
    borderRadius: 18,
    border: '1px solid #2A292F',
    padding: '20px 22px',
    boxShadow: '0 18px 40px rgba(0, 0, 0, 0.18)',
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: 16,
    flexWrap: 'wrap',
  },
  heroTitle: { fontSize: 28, fontWeight: 700, lineHeight: 1.08, color: '#ECEEF3' },
  heroSub: { fontSize: 12, color: '#8A90A7', lineHeight: 1.65, marginTop: 8, maxWidth: 780 },
  badge: (color) => ({
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    fontSize: 9,
    fontWeight: 700,
    padding: '5px 10px',
    borderRadius: 999,
    background: `${color}18`,
    color,
    border: `1px solid ${color}33`,
  }),
  workspace: {
    margin: '0 16px',
    background: 'linear-gradient(180deg, rgba(26, 25, 30, 0.98), rgba(20, 20, 24, 0.96))',
    borderRadius: 16,
    border: '1px solid #2A292F',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    boxShadow: '0 18px 40px rgba(0, 0, 0, 0.18)',
  },
  workspaceHeader: {
    padding: '18px 20px 14px',
    borderBottom: '1px solid #2A292F',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    flexWrap: 'wrap',
  },
  workspaceTitle: { fontSize: 10, fontWeight: 700, letterSpacing: 1.4, textTransform: 'uppercase', color: '#A0A5B8' },
  tabBar: {
    display: 'inline-flex',
    gap: 4,
    padding: 5,
    background: '#141419',
    border: '1px solid #2A292F',
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
    background: active ? 'rgba(240, 120, 37, 0.14)' : 'transparent',
    color: active ? '#F07825' : '#A0A5B8',
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
    background: 'linear-gradient(180deg, rgba(29, 28, 34, 0.98), rgba(21, 21, 26, 0.96))',
    borderRadius: 14,
    border: '1px solid #2A292F',
    overflow: 'hidden',
  },
  cardHeader: {
    padding: '14px 16px',
    borderBottom: '1px solid #2A292F',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
    flexWrap: 'wrap',
  },
  cardTitle: { fontSize: 10, fontWeight: 700, letterSpacing: 1.4, textTransform: 'uppercase', color: '#A0A5B8' },
  cardBody: { padding: 16 },
  fieldGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
    gap: 10,
  },
  fieldBlock: {
    background: '#201F25',
    border: '1px solid #2A292F',
    borderRadius: 12,
    padding: '12px 14px',
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  fieldLabel: { fontSize: 9, letterSpacing: 1.2, textTransform: 'uppercase', color: '#6B7186' },
  input: {
    width: '100%',
    borderRadius: 10,
    border: '1px solid rgba(255,255,255,0.06)',
    background: 'rgba(255,255,255,0.02)',
    color: '#ECEEF3',
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: 12,
    padding: '10px 12px',
    outline: 'none',
  },
  helper: { fontSize: 10, color: '#7E849A', lineHeight: 1.5 },
  profileHero: {
    background: 'radial-gradient(circle at top, rgba(91, 159, 228, 0.16), transparent 55%), linear-gradient(180deg, rgba(34, 33, 39, 0.96), rgba(24, 23, 28, 0.96))',
    border: '1px solid #2A292F',
    borderRadius: 14,
    padding: 18,
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
    gap: 14,
    minHeight: 100,
  },
  profileMeta: { fontSize: 12, color: '#8A90A7', lineHeight: 1.6 },
  sourceStack: { display: 'flex', flexDirection: 'column', gap: 12 },
  sourceCard: {
    background: 'linear-gradient(180deg, rgba(33, 32, 39, 0.96), rgba(23, 22, 27, 0.96))',
    border: '1px solid #2A292F',
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
  sourceName: { fontSize: 13, fontWeight: 700, color: '#ECEEF3' },
  sourceDesc: { fontSize: 11, color: '#8A90A7', lineHeight: 1.6 },
  sourceMeta: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, minmax(0, 1fr))',
    gap: 10,
  },
  smallField: {
    background: '#1F1E24',
    border: '1px solid rgba(255,255,255,0.04)',
    borderRadius: 12,
    padding: '10px 12px',
  },
  smallLabel: { fontSize: 9, color: '#6B7186', letterSpacing: 1.1, textTransform: 'uppercase' },
  smallValue: { marginTop: 8, fontSize: 12, color: '#ECEEF3', lineHeight: 1.5, wordBreak: 'break-word' },
  actionBtn: {
    border: '1px solid rgba(240, 120, 37, 0.35)',
    background: 'rgba(240, 120, 37, 0.14)',
    color: '#F7A35B',
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

const SourceCard = ({ source }) => (
  <article style={S.sourceCard}>
    <div style={S.sourceHead}>
      <div>
        <div style={S.sourceName}>{source.name}</div>
        <div style={S.sourceDesc}>{source.description}</div>
      </div>
      <span style={S.badge(source.statusColor)}>{source.status}</span>
    </div>
    <div style={S.sourceMeta}>
      <div style={S.smallField}>
        <div style={S.smallLabel}>Type</div>
        <div style={S.smallValue}>{source.type}</div>
      </div>
      <div style={S.smallField}>
        <div style={S.smallLabel}>Endpoint</div>
        <div style={S.smallValue}>{source.endpoint}</div>
      </div>
      <div style={S.smallField}>
        <div style={S.smallLabel}>Auth</div>
        <div style={S.smallValue}>{source.auth}</div>
      </div>
      <div style={S.smallField}>
        <div style={S.smallLabel}>Last Sync</div>
        <div style={S.smallValue}>{source.lastSync}</div>
      </div>
    </div>
  </article>
);

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

  const derivedSources = useMemo(() => ([
    {
      name: 'Forecast API',
      description: 'Primary service for config, settings, live forecast, and operational metadata.',
      type: 'REST API',
      endpoint: `${apiBaseUrl || '--'}/v2`,
      auth: 'Bearer / Session',
      lastSync: config?.partial_latest_date || effectiveDate || '--',
      status: 'Connected',
      statusColor: '#34D399',
    },
    {
      name: 'Day-Ahead Input Feed',
      description: 'Date-driven load and weather payload used for analysis, optimizer, and simulator pages.',
      type: 'Structured Dataset',
      endpoint: effectiveDate || '--',
      auth: 'Internal',
      lastSync: config?.latest_date || '--',
      status: effectiveDate ? 'Ready' : 'Waiting',
      statusColor: effectiveDate ? '#5B9FE4' : '#FBBF24',
    },
    {
      name: 'Actuals / Intraday Stream',
      description: 'Partial-day actuals and refresh feed used for forecast monitoring and live health signals.',
      type: 'Streaming / Incremental',
      endpoint: config?.partial_latest_date || '--',
      auth: 'Internal',
      lastSync: config?.partial_latest_date || 'Not available',
      status: config?.partial_latest_date ? 'Streaming' : 'Standby',
      statusColor: config?.partial_latest_date ? '#F07825' : '#6B7186',
    },
  ]), [apiBaseUrl, config?.latest_date, config?.partial_latest_date, effectiveDate]);

  const [sources, setSources] = useState(derivedSources);
  useEffect(() => setSources(derivedSources), [derivedSources]);

  const [newSource, setNewSource] = useState({
    name: '',
    type: 'REST API',
    endpoint: '',
    auth: 'Bearer / Session',
    description: '',
  });

  const [databases, setDatabases] = useState([
    {
      id: 'mysql',
      label: 'MySQL',
      host: 'mysql.internal.company',
      port: '3306',
      database: 'forecast_ops',
      schema: 'public',
      ssl: 'required',
      status: 'Available',
      statusColor: '#5B9FE4',
      description: 'Operational source for structured marts, tariff tables, and transaction-style histories.',
    },
    {
      id: 'postgres',
      label: 'PostgreSQL',
      host: 'postgres.analytics.company',
      port: '5432',
      database: 'load_analytics',
      schema: 'forecast',
      ssl: 'required',
      status: 'Available',
      statusColor: '#34D399',
      description: 'Analytics warehouse target for model outputs, historical backfills, and feature stores.',
    },
  ]);

  const handleProfileChange = (key, value) => {
    setProfile((prev) => ({ ...prev, [key]: value }));
  };

  const handleSourceChange = (key, value) => {
    setNewSource((prev) => ({ ...prev, [key]: value }));
  };

  const handleAddSource = () => {
    const name = String(newSource.name || '').trim();
    const endpoint = String(newSource.endpoint || '').trim();
    if (!name || !endpoint) return;
    setSources((prev) => ([
      ...prev,
      {
        name,
        description: String(newSource.description || '').trim() || 'User-added source configuration.',
        type: newSource.type,
        endpoint,
        auth: newSource.auth,
        lastSync: 'Pending test',
        status: 'Draft',
        statusColor: '#FBBF24',
      },
    ]));
    setNewSource({
      name: '',
      type: 'REST API',
      endpoint: '',
      auth: 'Bearer / Session',
      description: '',
    });
  };

  const handleDatabaseChange = (id, key, value) => {
    setDatabases((prev) => prev.map((db) => (
      db.id === id ? { ...db, [key]: value } : db
    )));
  };

  return (
    <main style={S.page}>
      <section style={S.hero}>
        <div>
          <div style={S.heroTitle}>Settings</div>
          <div style={S.heroSub}>
            Manage user identity, connected sources, database endpoints, and workspace defaults through tabs so each settings area is focused and usable.
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <span style={S.badge('#34D399')}><Shield size={12} /> Secure Session</span>
          <span style={S.badge('#5B9FE4')}><UserCircle2 size={12} /> {profile.role}</span>
          <span style={S.badge('#F07825')}><Globe size={12} /> {profile.region}</span>
        </div>
      </section>

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
                  <UserCircle2 size={44} color="#5B9FE4" />
                  <div style={{ marginTop: 14, fontSize: 22, fontWeight: 700 }}>{profile.name}</div>
                  <div style={S.profileMeta}>{profile.email}</div>
                  <div style={{ ...S.profileMeta, marginTop: 10 }}>{profile.team}</div>
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <span style={S.badge('#34D399')}>{profile.role}</span>
                  <span style={S.badge('#5B9FE4')}>{profile.timezone}</span>
                </div>
              </div>

              <div style={S.card}>
                <div style={S.cardHeader}>
                  <div style={S.cardTitle}>Profile Details</div>
                  <span style={S.badge('#5B9FE4')}><UserCircle2 size={12} /> Editable</span>
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

          {activeTab === 'sources' && (
            <div style={S.grid2}>
              <div style={S.card}>
                <div style={S.cardHeader}>
                  <div style={S.cardTitle}>Connected Sources</div>
                  <span style={S.badge('#34D399')}>{sources.length} linked</span>
                </div>
                <div style={S.cardBody}>
                  <div style={S.sourceStack}>
                    {sources.map((source) => <SourceCard key={`${source.name}-${source.endpoint}`} source={source} />)}
                  </div>
                </div>
              </div>

              <div style={S.card}>
                <div style={S.cardHeader}>
                  <div style={S.cardTitle}>Add Data Source</div>
                  <span style={S.badge('#F07825')}><Plus size={12} /> New Link</span>
                </div>
                <div style={S.cardBody}>
                  <div style={S.fieldGrid}>
                    <div style={S.fieldBlock}>
                      <label style={S.fieldLabel}>Source Name</label>
                      <input style={S.input} value={newSource.name} onChange={(e) => handleSourceChange('name', e.target.value)} placeholder="SCADA Actuals" />
                    </div>
                    <div style={S.fieldBlock}>
                      <label style={S.fieldLabel}>Type</label>
                      <input style={S.input} value={newSource.type} onChange={(e) => handleSourceChange('type', e.target.value)} />
                    </div>
                    <div style={S.fieldBlock}>
                      <label style={S.fieldLabel}>Endpoint / Path</label>
                      <input style={S.input} value={newSource.endpoint} onChange={(e) => handleSourceChange('endpoint', e.target.value)} placeholder="https://source.example/api" />
                    </div>
                    <div style={S.fieldBlock}>
                      <label style={S.fieldLabel}>Authentication</label>
                      <input style={S.input} value={newSource.auth} onChange={(e) => handleSourceChange('auth', e.target.value)} />
                    </div>
                    <div style={{ ...S.fieldBlock, gridColumn: '1 / -1' }}>
                      <label style={S.fieldLabel}>Description</label>
                      <textarea style={{ ...S.input, minHeight: 104, resize: 'vertical' }} value={newSource.description} onChange={(e) => handleSourceChange('description', e.target.value)} placeholder="What this source provides and where it is used." />
                      <div style={S.helper}>Adding a source updates the connected source list on this page immediately.</div>
                    </div>
                  </div>
                  <div style={{ marginTop: 14 }}>
                    <button type="button" style={S.actionBtn} onClick={handleAddSource}><Plus size={14} />Add Data Source</button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'databases' && (
            <div style={S.gridAuto}>
              {databases.map((db) => (
                <div key={db.id} style={S.card}>
                  <div style={S.cardHeader}>
                    <div style={S.cardTitle}>{db.label}</div>
                    <span style={S.badge(db.statusColor)}><Database size={12} /> {db.status}</span>
                  </div>
                  <div style={S.cardBody}>
                    <div style={{ ...S.helper, marginBottom: 12 }}>{db.description}</div>
                    <div style={S.fieldGrid}>
                      <div style={S.fieldBlock}>
                        <label style={S.fieldLabel}>Host</label>
                        <input style={S.input} value={db.host} onChange={(e) => handleDatabaseChange(db.id, 'host', e.target.value)} />
                      </div>
                      <div style={S.fieldBlock}>
                        <label style={S.fieldLabel}>Port</label>
                        <input style={S.input} value={db.port} onChange={(e) => handleDatabaseChange(db.id, 'port', e.target.value)} />
                      </div>
                      <div style={S.fieldBlock}>
                        <label style={S.fieldLabel}>Database</label>
                        <input style={S.input} value={db.database} onChange={(e) => handleDatabaseChange(db.id, 'database', e.target.value)} />
                      </div>
                      <div style={S.fieldBlock}>
                        <label style={S.fieldLabel}>Schema</label>
                        <input style={S.input} value={db.schema} onChange={(e) => handleDatabaseChange(db.id, 'schema', e.target.value)} />
                      </div>
                      <div style={S.fieldBlock}>
                        <label style={S.fieldLabel}>SSL Mode</label>
                        <input style={S.input} value={db.ssl} onChange={(e) => handleDatabaseChange(db.id, 'ssl', e.target.value)} />
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'workspace' && (
            <div style={S.gridAuto}>
              <div style={S.card}>
                <div style={S.cardHeader}>
                  <div style={S.cardTitle}>General Defaults</div>
                  <span style={S.badge('#F07825')}><Workflow size={12} /> Workspace</span>
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
                  <span style={S.badge('#5B9FE4')}><Globe size={12} /> API</span>
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
            </div>
          )}
        </div>
      </section>
    </main>
  );
}
