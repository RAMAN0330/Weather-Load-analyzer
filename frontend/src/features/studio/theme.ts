/**
 * Forecast Studio theme state (studio subtree only — legacy pages stay dark).
 *
 * - `mode` persists to localStorage ('vp-studio-theme'); default 'system'.
 * - Persisted value is read synchronously at store creation (localStorage is a
 *   sync storage, so zustand's persist hydrates before first render), and the
 *   system preference is read synchronously via matchMedia → no theme flash.
 */
import { useSyncExternalStore } from 'react';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import { cn } from '../../lib/utils';

export type ThemeMode = 'light' | 'dark' | 'system';
export type ResolvedTheme = 'light' | 'dark';

interface ThemeState {
  mode: ThemeMode;
  setMode: (m: ThemeMode) => void;
}

export const useStudioTheme = create<ThemeState>()(
  persist(
    (set) => ({
      mode: 'system',
      setMode: (mode) => set({ mode }),
    }),
    {
      name: 'vp-studio-theme',
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({ mode: s.mode }),
      version: 1,
    }
  )
);

// ── System preference (live) ─────────────────────────────────────────────────

const QUERY = '(prefers-color-scheme: dark)';
const mql: MediaQueryList | null =
  typeof window !== 'undefined' && typeof window.matchMedia === 'function'
    ? window.matchMedia(QUERY)
    : null;

function subscribeSystem(cb: () => void) {
  if (!mql) return () => {};
  if (typeof mql.addEventListener === 'function') {
    mql.addEventListener('change', cb);
    return () => mql.removeEventListener('change', cb);
  }
  // Safari < 14
  mql.addListener(cb);
  return () => mql.removeListener(cb);
}

const getSystemDark = () => (mql ? mql.matches : true);

export function useSystemPrefersDark(): boolean {
  return useSyncExternalStore(subscribeSystem, getSystemDark, () => true);
}

export function resolveTheme(mode: ThemeMode, systemDark: boolean): ResolvedTheme {
  if (mode === 'system') return systemDark ? 'dark' : 'light';
  return mode;
}

export function useThemeMode(): [ThemeMode, (m: ThemeMode) => void] {
  const mode = useStudioTheme((s) => s.mode);
  const setMode = useStudioTheme((s) => s.setMode);
  return [mode, setMode];
}

export function useResolvedTheme(): ResolvedTheme {
  const mode = useStudioTheme((s) => s.mode);
  const systemDark = useSystemPrefersDark();
  return resolveTheme(mode, systemDark);
}

/**
 * Class list for the studio root and every Radix portal content element
 * (portals render outside the studio subtree and must re-declare the scope).
 */
export function useThemeScopeClass(): string {
  const resolved = useResolvedTheme();
  return cn('ds-scope', resolved === 'light' ? 'theme-light' : 'theme-dark');
}
