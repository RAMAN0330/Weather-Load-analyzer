import type { ReactNode } from 'react';
import { AlertOctagon, RotateCw } from 'lucide-react';
import { Button } from '../../../components/ui/button';
import { Skeleton } from '../../../components/ui/skeleton';
import { cn } from '../../../lib/utils';

export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
}: {
  icon?: ReactNode;
  title: string;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        'data-surface flex min-h-0 flex-col items-center justify-center gap-3 overflow-hidden rounded-xl px-6 py-6 text-center',
        className
      )}
    >
      {icon && (
        <div className="grid size-12 place-items-center rounded-full border border-border bg-elev2 text-accent [&_svg]:size-5">
          {icon}
        </div>
      )}
      <div className="space-y-1">
        <h3 className="font-display text-base font-semibold text-fg">{title}</h3>
        {description && <p className="mx-auto max-w-md text-sm text-muted">{description}</p>}
      </div>
      {action}
    </div>
  );
}

export function ErrorState({
  title = 'Something went wrong',
  message,
  onRetry,
  retrying,
  className,
}: {
  title?: string;
  message: string;
  onRetry?: () => void;
  retrying?: boolean;
  className?: string;
}) {
  return (
    <div
      role="alert"
      className={cn(
        'flex min-h-0 flex-col items-center justify-center gap-3 overflow-hidden rounded-xl border border-danger/40 bg-danger/5 px-6 py-6 text-center',
        className
      )}
    >
      <AlertOctagon className="size-6 text-danger" aria-hidden />
      <div className="space-y-1">
        <h3 className="text-sm font-semibold text-fg">{title}</h3>
        <p className="mx-auto max-w-lg font-mono text-xs text-fg/80">{message}</p>
      </div>
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry} disabled={retrying}>
          <RotateCw className={cn(retrying && 'animate-spin')} aria-hidden />
          Retry
        </Button>
      )}
    </div>
  );
}

export function KpiSkeletonRow() {
  return (
    <div className="grid shrink-0 grid-cols-[repeat(auto-fit,minmax(128px,1fr))] gap-2" aria-hidden>
      {Array.from({ length: 7 }).map((_, i) => (
        <div
          key={i}
          className="glass-chrome flex h-[68px] flex-col justify-between rounded-lg px-3 py-2"
        >
          <Skeleton className="h-2.5 w-16" />
          <Skeleton className="h-5 w-24" />
          <Skeleton className="h-2.5 w-14" />
        </div>
      ))}
    </div>
  );
}

/** Fills its parent (h-full) — used inside viewport-fit panels. */
export function ChartSkeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn('data-surface flex h-full min-h-0 flex-col rounded-xl p-3', className)}
      aria-hidden
    >
      <div className="mb-3 flex items-center justify-between">
        <Skeleton className="h-3.5 w-40" />
        <Skeleton className="h-3.5 w-56" />
      </div>
      <Skeleton className="min-h-0 w-full flex-1" />
    </div>
  );
}
