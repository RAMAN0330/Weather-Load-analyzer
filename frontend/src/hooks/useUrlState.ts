import { useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useUi } from '../store/ui';
import type { Horizon } from '../types/api';

/** Two-way sync between the URL `?date=&horizon=&region=` and the UI store.
    Any page can render the current selections directly from the store; this
    hook keeps the URL in lockstep so the demo's deep links work. */
export function useUrlState() {
  const [params, setParams] = useSearchParams();
  const date = useUi((s) => s.date);
  const horizon = useUi((s) => s.horizon);
  const region = useUi((s) => s.region);
  const setDate = useUi((s) => s.setDate);
  const setHorizon = useUi((s) => s.setHorizon);
  const setRegion = useUi((s) => s.setRegion);

  // URL → store (run once on mount)
  useEffect(() => {
    const p = params;
    const urlDate = p.get('date');
    const urlHorizon = p.get('horizon') as Horizon | null;
    const urlRegion = p.get('region');
    if (urlDate && urlDate !== date) setDate(urlDate);
    if (urlHorizon && (urlHorizon === 't1' || urlHorizon === 't2') && urlHorizon !== horizon)
      setHorizon(urlHorizon);
    if (urlRegion && urlRegion !== region) setRegion(urlRegion);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Store → URL (replace, not push, so back-button doesn't fill up with toggles)
  useEffect(() => {
    if (!date && !horizon && !region) return;
    const next = new URLSearchParams(params);
    if (date) next.set('date', date);
    if (horizon) next.set('horizon', horizon);
    if (region) next.set('region', region);
    if (next.toString() !== params.toString()) {
      setParams(next, { replace: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [date, horizon, region]);
}
