import React, { useEffect, useState, lazy, Suspense } from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'
// Applies the persisted theme to <html> before first render (no flash).
import './features/studio/theme'
import { ErrorBoundary } from './ErrorBoundary'
import { useAuthStore } from './features/auth/authStore'
import LoginPage from './features/auth/LoginPage'

const App           = lazy(() => import('./App.jsx'))
const LandingPage   = lazy(() => import('./features/home/LandingPage.jsx'))
const AnalyticsShell = lazy(() => import('./features/analytics/AnalyticsShell.jsx'))

const SESSION_KEY = 'vp-mode'

function Spinner() {
  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ color: 'var(--accent)', fontFamily: "'IBM Plex Mono',monospace", fontSize: 13 }}>
        Forecast Studio…
      </div>
    </div>
  )
}

function AuthGate() {
  const token          = useAuthStore((s) => s.token)
  const user           = useAuthStore((s) => s.user)
  const logout         = useAuthStore((s) => s.logout)
  const rehydrateAxios = useAuthStore((s) => s.rehydrateAxios)
  const fetchMe        = useAuthStore((s) => s.fetchMe)

  const [ready, setReady] = useState(false)
  // Persist mode across refreshes within the same session
  const [mode, setMode] = useState(() => {
    try { return sessionStorage.getItem(SESSION_KEY) || null } catch { return null }
  })

  useEffect(() => {
    rehydrateAxios()
    fetchMe().finally(() => setReady(true))
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  if (!ready) return <Spinner />

  if (!token) {
    return <LoginPage onAuth={() => {}} />
  }

  const selectMode = (m) => {
    try { sessionStorage.setItem(SESSION_KEY, m) } catch { /* noop */ }
    setMode(m)
  }

  const goHome = () => {
    try { sessionStorage.removeItem(SESSION_KEY) } catch { /* noop */ }
    setMode(null)
  }

  // Landing page — first visit after login or explicit home navigation
  if (!mode) {
    return (
      <Suspense fallback={<Spinner />}>
        <LandingPage onSelect={selectMode} />
      </Suspense>
    )
  }

  if (mode === 'analytics') {
    return (
      <Suspense fallback={<Spinner />}>
        <AnalyticsShell onHome={goHome} />
      </Suspense>
    )
  }

  // mode === 'forecast' — original app
  return (
    <Suspense fallback={<Spinner />}>
      <App authUser={user} onLogout={logout} onHome={goHome} />
    </Suspense>
  )
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ErrorBoundary>
      <AuthGate />
    </ErrorBoundary>
  </React.StrictMode>,
)
