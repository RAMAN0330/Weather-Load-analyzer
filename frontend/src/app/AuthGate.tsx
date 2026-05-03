import { useEffect, useState } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '../features/auth/authStore';
import { Skeleton } from '../components/ui/skeleton';

export function AuthGate({ children }: { children: React.ReactNode }) {
  const token = useAuthStore((s: any) => s.token);
  const fetchMe = useAuthStore((s: any) => s.fetchMe);
  const [checking, setChecking] = useState(Boolean(token));
  const location = useLocation();

  useEffect(() => {
    if (!token) {
      setChecking(false);
      return;
    }
    let cancelled = false;
    fetchMe().finally(() => {
      if (!cancelled) setChecking(false);
    });
    return () => {
      cancelled = true;
    };
  }, [token, fetchMe]);

  if (checking) {
    return (
      <div className="flex h-screen flex-col gap-3 p-6">
        <Skeleton className="h-9 w-64" />
        <Skeleton className="h-9 w-80" />
        <div className="mt-4 grid flex-1 grid-cols-4 gap-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-32" />
          ))}
        </div>
      </div>
    );
  }

  const isAuthed = useAuthStore.getState().token != null;
  if (!isAuthed) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  return <>{children}</>;
}
