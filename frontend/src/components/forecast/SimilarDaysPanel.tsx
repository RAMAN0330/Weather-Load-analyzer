import { Calendar } from 'lucide-react';
import { Badge } from '../ui/badge';
import { Card, CardContent, CardHeader, CardTitle, CardSubtitle } from '../ui/card';
import { Skeleton } from '../ui/skeleton';
import type { SimilarDayPick } from '../../types/api';
import { dayType, fmtNum } from '../../lib/utils';

interface Props {
  picks: SimilarDayPick[] | undefined;
  loading?: boolean;
  compact?: boolean;
}

export function SimilarDaysPanel({ picks, loading, compact }: Props) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Similar days</CardTitle>
            <CardSubtitle>{picks?.length ? `${picks.length} matched` : 'How the baseline was built'}</CardSubtitle>
          </div>
          <Calendar className="h-4 w-4 text-accent" />
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="flex flex-col gap-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-9" />
            ))}
          </div>
        ) : picks && picks.length ? (
          <ul className="flex flex-col divide-y divide-border">
            {picks.slice(0, compact ? 6 : 12).map((p) => (
              <li key={p.date} className="flex items-center justify-between gap-3 py-2">
                <div className="flex flex-col">
                  <span className="font-mono text-sm font-semibold text-fg">{p.date}</span>
                  <span className="text-2xs uppercase tracking-wider text-faint">
                    {dayType(p.date)}
                    {p.rain_match != null && (
                      <span className="ml-2">
                        {p.rain_match ? '· rain match' : '· dry'}
                      </span>
                    )}
                  </span>
                </div>
                <div className="flex flex-col items-end gap-1">
                  {p.scale != null && (
                    <Badge variant={Math.abs(p.scale - 1) > 0.1 ? 'warning' : 'default'}>
                      ×{fmtNum(p.scale, 2)}
                    </Badge>
                  )}
                  {p.similarity_score != null && (
                    <span className="font-mono text-2xs text-faint">
                      score {fmtNum(p.similarity_score, 2)}
                    </span>
                  )}
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <div className="py-6 text-center text-sm text-muted">No similar days returned.</div>
        )}
      </CardContent>
    </Card>
  );
}
