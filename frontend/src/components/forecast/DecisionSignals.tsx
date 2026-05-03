import { AlertTriangle, AlertCircle, Info } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardSubtitle } from '../ui/card';
import { Skeleton } from '../ui/skeleton';
import { Badge } from '../ui/badge';
import type { ForecastResponse } from '../../types/api';
import { blockToTime } from '../../lib/utils';

interface Props {
  data: ForecastResponse | undefined;
  loading?: boolean;
}

const ICONS: Record<string, any> = {
  high: AlertTriangle,
  medium: AlertCircle,
  low: Info,
};

const VARIANTS: Record<string, 'danger' | 'warning' | 'info'> = {
  high: 'danger',
  medium: 'warning',
  low: 'info',
};

export function DecisionSignals({ data, loading }: Props) {
  const signals = (data?.decision_signals || []) as Array<{
    block?: number;
    severity?: string;
    message?: string;
  }>;
  const sorted = [...signals].sort((a, b) => {
    const order = { high: 0, medium: 1, low: 2 } as Record<string, number>;
    return (order[(a.severity || 'low').toLowerCase()] ?? 3) - (order[(b.severity || 'low').toLowerCase()] ?? 3);
  });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Decision signals</CardTitle>
        <CardSubtitle>Risks the model wants you to know about</CardSubtitle>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="flex flex-col gap-2">
            <Skeleton className="h-12" />
            <Skeleton className="h-12" />
          </div>
        ) : sorted.length ? (
          <ul className="flex flex-col gap-2">
            {sorted.slice(0, 6).map((s, i) => {
              const sev = (s.severity || 'low').toLowerCase();
              const Icon = ICONS[sev] || Info;
              return (
                <li
                  key={i}
                  className="flex items-start gap-3 rounded-md border border-border bg-elev2/40 p-3"
                >
                  <div
                    className={`grid h-7 w-7 shrink-0 place-items-center rounded-md bg-${VARIANTS[sev]}/15 text-${VARIANTS[sev]}`}
                  >
                    <Icon className="h-3.5 w-3.5" />
                  </div>
                  <div className="flex flex-1 flex-col gap-1">
                    <div className="flex items-center gap-2">
                      <Badge variant={VARIANTS[sev] || 'default'}>{sev}</Badge>
                      {s.block != null && (
                        <span className="font-mono text-2xs text-muted">
                          block {s.block} · {blockToTime(s.block)}
                        </span>
                      )}
                    </div>
                    <div className="text-sm text-fg">{s.message || '—'}</div>
                  </div>
                </li>
              );
            })}
          </ul>
        ) : (
          <div className="py-6 text-center text-sm text-muted">All clear · no signals raised.</div>
        )}
      </CardContent>
    </Card>
  );
}
