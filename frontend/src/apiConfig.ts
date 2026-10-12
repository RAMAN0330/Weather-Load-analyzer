/**
 * Centralized API configuration for the frontend.
 * This ensures consistency across all services and components.
 */

const normalizeBase = (base: string, fallback = '/api', appendApi = true): string => {
  if (!base) return fallback;
  const trimmed = base.replace(/\/$/, '');
  return appendApi && !trimmed.endsWith('/api') ? `${trimmed}/api` : trimmed;
};

/**
 * The base URL for app/API requests (FastAPI, same origin via nginx / Vite proxy).
 * Priority:
 * 1. import.meta.env.VITE_API_BASE_URL
 * 2. '/api' (relative path, works with Vite proxy in dev and same-origin in prod)
 */
export const API_BASE = normalizeBase(
  import.meta.env.VITE_API_BASE_URL || '',
  '/api',
  true
);

/**
 * '/ml-api/*' is an alias of FastAPI '/api/*' (rewritten by Vite / nginx).
 */
const fastApiBaseOverride = import.meta.env.VITE_FASTAPI_API_BASE_URL || '';

export const TRAINING_API_BASE = fastApiBaseOverride
  ? normalizeBase(fastApiBaseOverride, '/ml-api', !fastApiBaseOverride.startsWith('/'))
  : '/ml-api';

/**
 * Helper to build a full API URL given a sub-path.
 */
export const getApiUrl = (path: string): string => {
  const base = API_BASE.replace(/\/$/, '');
  const sub = path.startsWith('/') ? path : `/${path}`;
  return `${base}${sub}`;
};

export const getTrainingApiUrl = (path: string): string => {
  const base = TRAINING_API_BASE.replace(/\/$/, '');
  const sub = path.startsWith('/') ? path : `/${path}`;
  return `${base}${sub}`;
};
