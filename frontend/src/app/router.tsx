import { lazy, Suspense } from 'react';
import { createBrowserRouter, Navigate, RouterProvider } from 'react-router-dom';
import { AppShell } from './layout/AppShell';
import { AuthGate } from './AuthGate';
import { Skeleton } from '../components/ui/skeleton';

const ForecastPage = lazy(() => import('../features/forecast/ForecastPage'));
const WeatherPage = lazy(() => import('../features/weather/WeatherPage'));
const MonitorPage = lazy(() => import('../features/monitor/MonitorPage'));
const SimulatorPage = lazy(() => import('../features/simulator/SimulatorPage'));
const BacktestPage = lazy(() => import('../features/backtest/BacktestPage'));
const SettingsPage = lazy(() => import('../features/settings/SettingsPage'));
const LoginPage = lazy(() => import('../features/auth/LoginPage'));

function PageFallback() {
  return (
    <div className="flex h-full flex-col gap-4 p-6">
      <Skeleton className="h-8 w-48" />
      <div className="grid grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-24" />
        ))}
      </div>
      <Skeleton className="flex-1 min-h-[320px]" />
    </div>
  );
}

const wrap = (el: React.ReactNode) => <Suspense fallback={<PageFallback />}>{el}</Suspense>;

const router = createBrowserRouter([
  {
    path: '/login',
    element: wrap(<LoginPage />),
  },
  {
    path: '/',
    element: (
      <AuthGate>
        <AppShell />
      </AuthGate>
    ),
    children: [
      { index: true, element: <Navigate to="/forecast" replace /> },
      { path: 'forecast', element: wrap(<ForecastPage />) },
      { path: 'weather', element: wrap(<WeatherPage />) },
      { path: 'monitor', element: wrap(<MonitorPage />) },
      { path: 'simulator', element: wrap(<SimulatorPage />) },
      { path: 'backtest', element: wrap(<BacktestPage />) },
      { path: 'settings', element: wrap(<SettingsPage />) },
      { path: '*', element: <Navigate to="/forecast" replace /> },
    ],
  },
]);

export function Router() {
  return <RouterProvider router={router} />;
}
