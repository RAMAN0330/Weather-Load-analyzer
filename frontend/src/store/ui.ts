import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { Horizon } from '../types/api';

interface UiState {
  date: string;          // YYYY-MM-DD (T+1 target / "current selected day")
  horizon: Horizon;      // 't1' | 't2'
  region: string;        // 'haryana' | 'odisha' | ...
  sidebarCollapsed: boolean;
  baselineDays: number;
  setDate: (d: string) => void;
  setHorizon: (h: Horizon) => void;
  setRegion: (r: string) => void;
  setBaselineDays: (n: number) => void;
  toggleSidebar: () => void;
  hydrateFromConfig: (cfg: { default_date?: string; default_region?: string; best_baseline_window?: number }) => void;
}

export const useUi = create<UiState>()(
  persist(
    (set, get) => ({
      date: '',
      horizon: 't1',
      region: 'haryana',
      sidebarCollapsed: false,
      baselineDays: 7,
      setDate: (d) => set({ date: d }),
      setHorizon: (h) => set({ horizon: h }),
      setRegion: (r) => set({ region: r }),
      setBaselineDays: (n) => set({ baselineDays: Math.max(1, Math.min(30, n)) }),
      toggleSidebar: () => set({ sidebarCollapsed: !get().sidebarCollapsed }),
      hydrateFromConfig: (cfg) =>
        set((s) => ({
          date: s.date || cfg.default_date || '',
          region: s.region || cfg.default_region || 'haryana',
          baselineDays: s.baselineDays || cfg.best_baseline_window || 7,
        })),
    }),
    {
      name: 'vp-ui',
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({
        region: s.region,
        baselineDays: s.baselineDays,
        sidebarCollapsed: s.sidebarCollapsed,
      }),
    }
  )
);
