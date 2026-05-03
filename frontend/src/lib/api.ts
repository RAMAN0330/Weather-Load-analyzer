import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { getApiUrl, getTrainingApiUrl } from '../apiConfig';

/** Single shared axios instance.  Auth interceptors live in authStore.js
    and attach the Authorization header globally on the default axios. */
export const api: AxiosInstance = axios;

const FORECAST_TIMEOUT_MS = 5 * 60 * 1000; // 5 min

export interface FetchOpts extends AxiosRequestConfig {
  /** Override the default timeout in ms. */
  timeoutMs?: number;
}

function withTimeout(opts?: FetchOpts): AxiosRequestConfig {
  const { timeoutMs, ...rest } = opts || {};
  return { timeout: timeoutMs ?? 60_000, ...rest };
}

/** Django (`/api/v2/...`) */
export async function djangoPost<T>(path: string, body: unknown, opts?: FetchOpts): Promise<T> {
  const url = getApiUrl(`/v2${path}`);
  const res = await api.post(url, body, withTimeout(opts));
  return res.data as T;
}

export async function djangoGet<T>(path: string, opts?: FetchOpts): Promise<T> {
  const url = getApiUrl(`/v2${path}`);
  const res = await api.get(url, withTimeout(opts));
  return res.data as T;
}

/** FastAPI (`/ml-api/v2/...`) */
export async function fastapiPost<T>(path: string, body: unknown, opts?: FetchOpts): Promise<T> {
  const url = getTrainingApiUrl(`/v2${path}`);
  const res = await api.post(url, body, withTimeout({ timeoutMs: FORECAST_TIMEOUT_MS, ...opts }));
  return res.data as T;
}

export async function fastapiGet<T>(path: string, opts?: FetchOpts): Promise<T> {
  const url = getTrainingApiUrl(`/v2${path}`);
  const res = await api.get(url, withTimeout(opts));
  return res.data as T;
}
