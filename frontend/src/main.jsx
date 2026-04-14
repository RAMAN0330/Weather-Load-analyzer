import React, { useEffect, useState } from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import { ErrorBoundary } from './ErrorBoundary'
import { useAuthStore } from './features/auth/authStore'
import LoginPage from './features/auth/LoginPage'

/**
 * AuthGate — owns all auth logic outside <App> so App never has
 * conditional early-returns that violate Rules of Hooks.
 */
function AuthGate() {
  const token          = useAuthStore((s) => s.token)
  const user           = useAuthStore((s) => s.user)
  const logout         = useAuthStore((s) => s.logout)
  const rehydrateAxios = useAuthStore((s) => s.rehydrateAxios)
  const fetchMe        = useAuthStore((s) => s.fetchMe)

  // ready = false during the initial boot token-validation request
  const [ready, setReady] = useState(false)

  useEffect(() => {
    rehydrateAxios()          // re-attach Bearer header from sessionStorage
    fetchMe().finally(() => setReady(true))   // validate token; clear if stale
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Still checking session — show nothing (avoids flash of login page)
  if (!ready) {
    return (
      <div style={{
        minHeight: '100vh', background: '#0E0D12',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <div style={{ color: '#F07825', fontFamily: "'IBM Plex Mono',monospace", fontSize: 13 }}>
          VidyutPragya…
        </div>
      </div>
    )
  }

  // No valid session → show login/register
  if (!token) {
    return <LoginPage onAuth={() => {
      // After login the store already has token+user set.
      // Just re-validate to sync the user object, then gate will
      // re-render automatically because token subscription fires.
    }} />
  }

  // Authenticated → render full app, passing user info + logout down
  return <App authUser={user} onLogout={logout} />
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ErrorBoundary>
      <AuthGate />
    </ErrorBoundary>
  </React.StrictMode>,
)
