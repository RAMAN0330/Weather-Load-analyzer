import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import axios from 'axios';

// Django runs on port 8001 — auth lives entirely there.
const DJANGO_BASE = import.meta.env.VITE_DJANGO_URL
  ? import.meta.env.VITE_DJANGO_URL.replace(/\/$/, '')
  : 'http://localhost:8001';

const authUrl = (path) => `${DJANGO_BASE}/auth/${path}`;

// ── Axios 401 interceptor ─────────────────────────────────────────────────────
// Fires on any API 401 EXCEPT the auth/me validation call on boot
// (that one handles its own 401 inside fetchMe).
let _logoutFn = null;
let _suppressInterceptor = false;   // true while fetchMe boot-check is running

axios.interceptors.response.use(
  (res) => res,
  (err) => {
    if (!_suppressInterceptor && err?.response?.status === 401 && _logoutFn) {
      _logoutFn();
    }
    return Promise.reject(err);
  }
);

// ── Store ─────────────────────────────────────────────────────────────────────
export const useAuthStore = create(
  persist(
    (set, get) => ({
      user:  null,
      token: null,

      isAuthenticated: () => !!get().token,

      register: async (username, email, password) => {
        const res = await axios.post(authUrl('register'), { username, email, password });
        return res.data;
      },

      login: async (username, password) => {
        const res = await axios.post(authUrl('login'), { username, password });
        const { token, user } = res.data;
        set({ token, user });
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        return user;
      },

      logout: async () => {
        const { token } = get();
        if (token) {
          try {
            await axios.post(
              authUrl('logout'),
              {},
              { headers: { Authorization: `Bearer ${token}` }, timeout: 4000 }
            );
          } catch (_) { /* ignore */ }
        }
        delete axios.defaults.headers.common['Authorization'];
        set({ user: null, token: null });
      },

      // Re-attach token to axios on boot and wire the 401 interceptor
      rehydrateAxios: () => {
        const { token, logout } = get();
        _logoutFn = logout;
        if (token) {
          axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        }
      },

      // Validate stored token on boot — suppress the interceptor during this call
      // so a stale token doesn't trigger a double-logout.
      fetchMe: async () => {
        const { token } = get();
        if (!token) return null;
        _suppressInterceptor = true;
        try {
          const res = await axios.get(authUrl('me'), {
            headers: { Authorization: `Bearer ${token}` },
          });
          set({ user: res.data });
          return res.data;
        } catch (_) {
          // Token invalid / server down → clear silently
          delete axios.defaults.headers.common['Authorization'];
          set({ user: null, token: null });
          return null;
        } finally {
          _suppressInterceptor = false;
        }
      },
    }),
    {
      name: 'vp-auth',
      // sessionStorage → cleared when browser tab/window is closed
      storage: createJSONStorage(() => sessionStorage),
      partialize: (s) => ({ token: s.token, user: s.user }),
    }
  )
);
