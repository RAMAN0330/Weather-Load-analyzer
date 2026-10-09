import { useMemo } from 'react';
import { useResolvedTheme } from '../features/studio/theme';

/**
 * Resolve CSS custom properties (e.g. `--text`) to concrete colour strings for
 * libraries that can't take `var(...)` (SVG presentation attributes, canvas).
 * Recomputes when the app theme changes.
 */
export function useThemeColors<K extends string>(vars: Record<K, string>): Record<K, string> {
  const theme = useResolvedTheme();
  const key = JSON.stringify(vars);
  return useMemo(() => {
    const cs = typeof document !== 'undefined' ? getComputedStyle(document.documentElement) : null;
    const out = {} as Record<K, string>;
    (Object.keys(vars) as K[]).forEach((k) => {
      out[k] = (cs?.getPropertyValue(vars[k]).trim() || '') as string;
    });
    return out;
    // `theme` triggers re-reading after <html data-theme> changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme, key]);
}
