import { AlertTriangle, CheckCircle2, XCircle } from 'lucide-react';
import { Badge } from '../../../components/ui/badge';
import type { QualityStatus } from '../api';

const MAP: Record<QualityStatus, { variant: 'success' | 'warning' | 'danger'; label: string }> = {
  ok: { variant: 'success', label: 'OK' },
  degraded: { variant: 'warning', label: 'Degraded' },
  failed: { variant: 'danger', label: 'Failed' },
};

export function StatusBadge({
  status,
  className,
}: {
  status?: QualityStatus | null;
  className?: string;
}) {
  if (!status) return null;
  const s = MAP[status] ?? { variant: 'warning' as const, label: String(status) };
  const Icon = status === 'ok' ? CheckCircle2 : status === 'failed' ? XCircle : AlertTriangle;
  return (
    <Badge variant={s.variant} className={className} aria-label={`Status: ${s.label}`}>
      <Icon className="size-3" aria-hidden />
      {s.label}
    </Badge>
  );
}
