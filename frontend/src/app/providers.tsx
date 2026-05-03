import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { TooltipProvider } from '../components/ui/tooltip';
import { Toaster } from 'sonner';
import { useEffect } from 'react';
import { useAuthStore } from '../features/auth/authStore';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 60_000,
    },
  },
});

export function Providers({ children }: { children: React.ReactNode }) {
  // Re-attach token + 401 interceptor on mount.
  const rehydrate = useAuthStore((s: any) => s.rehydrateAxios);
  useEffect(() => {
    rehydrate?.();
  }, [rehydrate]);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider delayDuration={150}>
        {children}
        <Toaster
          theme="dark"
          richColors
          position="bottom-right"
          toastOptions={{
            style: {
              background: 'hsl(var(--bg-elev-2))',
              border: '1px solid hsl(var(--border))',
              color: 'hsl(var(--fg))',
            },
          }}
        />
        <ReactQueryDevtools initialIsOpen={false} />
      </TooltipProvider>
    </QueryClientProvider>
  );
}
