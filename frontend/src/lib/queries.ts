import { useQuery, useQueries, UseQueryOptions } from '@tanstack/react-query';
import { djangoGet, djangoPost, fastapiPost } from './api';
import type { ConfigResponse, ForecastResponse, Horizon } from '../types/api';

/** Centralised query keys — never construct strings inline elsewhere. */
export const qk = {
  config: () => ['config'] as const,
  live: (date: string, region: string) => ['live', date, region] as const,
  forecastT2: (date: string, region: string) => ['forecast-t2', date, region] as const,
  dayahead: (date: string, region: string) => ['dayahead', date, region] as const,
  weatherT1: (date: string, region: string) => ['weather-t1', date, region] as const,
  weatherT2: (date: string, region: string) => ['weather-t2', date, region] as const,
  analysis: (date: string, region: string) => ['analysis', date, region] as const,
};

const STALE_5_MIN = 5 * 60 * 1000;
const STALE_10_MIN = 10 * 60 * 1000;

export function useConfig(opts?: Partial<UseQueryOptions<ConfigResponse>>) {
  return useQuery<ConfigResponse>({
    queryKey: qk.config(),
    queryFn: () => djangoGet<ConfigResponse>('/config'),
    staleTime: STALE_10_MIN,
    ...opts,
  });
}

export function useLiveForecast(
  date: string | undefined,
  region: string,
  baselineDays = 7,
  opts?: Partial<UseQueryOptions<ForecastResponse>>
) {
  return useQuery<ForecastResponse>({
    queryKey: qk.live(date ?? '', region),
    queryFn: () =>
      fastapiPost<ForecastResponse>('/live', {
        date,
        region,
        baseline_days: baselineDays,
        actual_blocks: 0,
      }),
    enabled: Boolean(date && region),
    staleTime: STALE_5_MIN,
    retry: 1,
    ...opts,
  });
}

export function useT2Forecast(
  date: string | undefined,
  region: string,
  baselineDays = 7,
  opts?: Partial<UseQueryOptions<ForecastResponse>>
) {
  return useQuery<ForecastResponse>({
    queryKey: qk.forecastT2(date ?? '', region),
    queryFn: () =>
      fastapiPost<ForecastResponse>('/forecast/t2', {
        date,
        region,
        baseline_days: baselineDays,
      }),
    enabled: Boolean(date && region),
    staleTime: STALE_5_MIN,
    retry: 1,
    ...opts,
  });
}

export function useDayAhead(
  date: string | undefined,
  region: string,
  baselineDays = 7,
  opts?: Partial<UseQueryOptions<ForecastResponse>>
) {
  return useQuery<ForecastResponse>({
    queryKey: qk.dayahead(date ?? '', region),
    queryFn: () =>
      fastapiPost<ForecastResponse>('/dayahead', {
        date,
        region,
        baseline_days: baselineDays,
      }),
    enabled: Boolean(date && region),
    staleTime: STALE_5_MIN,
    retry: 1,
    ...opts,
  });
}

/** Convenience: T+1 + T+2 + T+1 dayahead in one shot, with the T+2-date
    dayahead chained off the T+2 response so the Weather page has both days
    available regardless of horizon. */
export function useForecastBundle(date: string | undefined, region: string, baselineDays = 7) {
  const live = useLiveForecast(date, region, baselineDays);
  const t2 = useT2Forecast(date, region, baselineDays);
  const t1Weather = useDayAhead(date, region, baselineDays, {
    enabled: Boolean(date && region),
  });
  const t2Date = t2.data?.t2_date;
  const t2Weather = useDayAhead(t2Date, region, baselineDays, {
    enabled: Boolean(t2Date && region),
  });

  return { live, t2, t1Weather, t2Weather };
}

export function pickByHorizon<T>(horizon: Horizon, t1: T | undefined, t2: T | undefined): T | undefined {
  return horizon === 't2' ? t2 : t1;
}
