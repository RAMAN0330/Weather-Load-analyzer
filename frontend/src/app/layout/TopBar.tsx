import { useEffect } from 'react';
import { Calendar, ChevronDown, RefreshCw, MapPin, PanelLeftClose, PanelLeft } from 'lucide-react';
import { Button } from '../../components/ui/button';
import { ToggleGroup, ToggleGroupItem } from '../../components/ui/toggle-group';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '../../components/ui/popover';
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from '../../components/ui/dropdown-menu';
import { Input } from '../../components/ui/input';
import { Skeleton } from '../../components/ui/skeleton';
import { useUi } from '../../store/ui';
import { useConfig } from '../../lib/queries';
import { addDays, dayType } from '../../lib/utils';
import { useQueryClient } from '@tanstack/react-query';

export function TopBar() {
  const { date, horizon, region, sidebarCollapsed, setDate, setHorizon, setRegion, toggleSidebar, hydrateFromConfig } =
    useUi();

  const { data: config, isLoading } = useConfig();
  const qc = useQueryClient();

  useEffect(() => {
    if (config) {
      hydrateFromConfig({
        default_date: config.default_date,
        default_region: config.default_region,
        best_baseline_window: config.best_baseline_window,
      });
    }
  }, [config, hydrateFromConfig]);

  const t2Date = date ? addDays(date, 1) : '';
  const regions = config?.available_regions || ['haryana', 'odisha', 'rajasthan'];

  const refresh = () => qc.invalidateQueries();

  return (
    <header className="sticky top-0 z-40 flex h-14 shrink-0 items-center gap-3 border-b border-border bg-base/80 px-4 backdrop-blur-md">
      <Button variant="ghost" size="icon" onClick={toggleSidebar} className="-ml-1">
        {sidebarCollapsed ? <PanelLeft className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
      </Button>

      {/* Region picker */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="outline" size="sm" className="gap-2 capitalize">
            <MapPin className="h-3.5 w-3.5 text-accent" />
            <span>{region}</span>
            <ChevronDown className="h-3 w-3 opacity-60" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          <DropdownMenuLabel>Region</DropdownMenuLabel>
          <DropdownMenuSeparator />
          {regions.map((r) => (
            <DropdownMenuItem key={r} onSelect={() => setRegion(r)} className="capitalize">
              {r}
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>

      {/* Date picker (simple — popover with input) */}
      <Popover>
        <PopoverTrigger asChild>
          <Button variant="outline" size="sm" className="gap-2 font-mono">
            <Calendar className="h-3.5 w-3.5 text-accent" />
            {isLoading && !date ? (
              <Skeleton className="h-4 w-24" />
            ) : (
              <>
                <span>{date || '—'}</span>
                <span className="text-faint">·</span>
                <span className="text-muted">{date ? dayType(date) : ''}</span>
              </>
            )}
            <ChevronDown className="h-3 w-3 opacity-60" />
          </Button>
        </PopoverTrigger>
        <PopoverContent align="start" className="w-64">
          <div className="flex flex-col gap-2">
            <label className="text-2xs font-semibold uppercase tracking-wider text-faint">
              Forecast date (T+1)
            </label>
            <Input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              max={config?.partial_latest_date || config?.latest_date}
            />
            <div className="text-xs text-muted">
              T+2 will be{' '}
              <span className="font-mono text-fg">{t2Date || '—'}</span>{' '}
              <span className="text-faint">({t2Date ? dayType(t2Date) : ''})</span>
            </div>
          </div>
        </PopoverContent>
      </Popover>

      {/* Horizon */}
      <ToggleGroup
        type="single"
        value={horizon}
        onValueChange={(v) => v && setHorizon(v as 't1' | 't2')}
      >
        <ToggleGroupItem value="t1">T+1</ToggleGroupItem>
        <ToggleGroupItem value="t2">T+2</ToggleGroupItem>
      </ToggleGroup>

      <div className="ml-auto flex items-center gap-2">
        <Button variant="ghost" size="sm" onClick={refresh} className="gap-2">
          <RefreshCw className="h-3.5 w-3.5" />
          <span>Refresh</span>
        </Button>
      </div>
    </header>
  );
}
