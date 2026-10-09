import { useState } from 'react';
import {
  Check,
  ChevronDown,
  Copy,
  Download,
  Info,
  Loader2,
  MapPin,
  Play,
  Sparkles,
} from 'lucide-react';
import { toast } from 'sonner';
import { Button } from '../../../components/ui/button';
import { Input } from '../../../components/ui/input';
import { Label } from '../../../components/ui/label';
import { ToggleGroup, ToggleGroupItem } from '../../../components/ui/toggle-group';
import { Popover, PopoverContent, PopoverTrigger } from '../../../components/ui/popover';
import { Tooltip, TooltipContent, TooltipTrigger } from '../../../components/ui/tooltip';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from '../../../components/ui/dropdown-menu';
import { cn } from '../../../lib/utils';
import type { ForecastResponse, Horizon, ModelInfo } from '../api';
import { useThemeScopeClass } from '../theme';
import { copyText, fmtDateShort, fmtDateTimeIST, shiftDate, titleCase } from '../utils';
import { StatusBadge } from './StatusBadge';

export interface CommandBarProps {
  region: string;
  regions: string[];
  onRegion: (r: string) => void;
  targetDate: string;
  onTargetDate: (d: string) => void;
  horizon: Horizon;
  onHorizon: (h: Horizon) => void;
  model: string;
  models: ModelInfo[];
  modelsFallback: boolean;
  onModel: (m: string) => void;
  onRun: () => void;
  running: boolean;
  dirty: boolean;
  forecast?: ForecastResponse;
  onExport: () => void;
}

const triggerCls =
  'inline-flex h-8 items-center justify-between gap-2 rounded-md border border-border bg-elev2 px-2.5 text-sm text-fg transition-colors hover:bg-elev3 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring';

/**
 * Single-row command bar at ≥1280px (title + status | controls | actions);
 * wraps to two rows below that. Field labels are visually hidden but present
 * for assistive tech; run metadata lives in the info popover.
 */
export function CommandBar(p: CommandBarProps) {
  const [copied, setCopied] = useState(false);
  const scope = useThemeScopeClass();
  const f = p.forecast;
  const modelLabel = p.models.find((m) => m.key === p.model)?.label ?? titleCase(p.model);
  const origin = p.targetDate ? shiftDate(p.targetDate, -p.horizon) : '';

  const onCopy = async () => {
    if (!f) return;
    const ok = await copyText(f.forecast_id);
    if (ok) {
      setCopied(true);
      toast.success('Forecast ID copied', { description: f.forecast_id });
      setTimeout(() => setCopied(false), 1500);
    } else {
      toast.error('Could not access clipboard');
    }
  };

  return (
    <header
      className="glass-strong relative z-20 shrink-0 rounded-xl px-3 py-2"
      aria-label="Forecast command bar"
    >
      <form
        className="flex flex-wrap items-center gap-x-3 gap-y-2 xl:flex-nowrap"
        onSubmit={(e) => {
          e.preventDefault();
          p.onRun();
        }}
      >
        {/* Identity + status */}
        <div className="flex shrink-0 items-center gap-2">
          <span className="grid size-7 shrink-0 place-items-center rounded-lg bg-accent/15 text-accent">
            <Sparkles className="size-3.5" aria-hidden />
          </span>
          <div className="leading-tight">
            <h1 className="whitespace-nowrap font-display text-sm font-semibold text-fg">
              Studio
            </h1>
            <p
              className="truncate font-mono text-2xs text-muted"
              title="Only data dated on or before the origin is used"
            >
              T+{p.horizon} · origin {origin ? fmtDateShort(origin) : '—'}
            </p>
          </div>
          {f && (
            <Popover>
              <Tooltip>
                <TooltipTrigger asChild>
                  <PopoverTrigger
                    className="inline-flex shrink-0 items-center gap-1 rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    aria-label={`Status ${f.status}. Run details: forecast ID, versions`}
                  >
                    <StatusBadge status={f.status} className="cursor-pointer" />
                    <Info className="size-3.5 text-muted" aria-hidden />
                  </PopoverTrigger>
                </TooltipTrigger>
                <TooltipContent className={scope}>Run details</TooltipContent>
              </Tooltip>
              <PopoverContent align="start" className={cn(scope, 'glass-strong w-80')}>
                <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-xs">
                  <dt className="text-muted">Forecast ID</dt>
                  <dd>
                    <button
                      type="button"
                      onClick={onCopy}
                      className="inline-flex max-w-full items-center gap-1.5 rounded border border-border bg-elev2 px-1.5 py-0.5 font-mono text-fg hover:bg-elev3 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      aria-label={`Copy forecast ID ${f.forecast_id}`}
                    >
                      <span className="truncate">{f.forecast_id}</span>
                      {copied ? (
                        <Check className="size-3 shrink-0 text-success" aria-hidden />
                      ) : (
                        <Copy className="size-3 shrink-0 text-muted" aria-hidden />
                      )}
                    </button>
                  </dd>
                  <dt className="text-muted">Model version</dt>
                  <dd className="font-mono">{f.model_version}</dd>
                  <dt className="text-muted">Feature version</dt>
                  <dd className="font-mono">{f.feature_version}</dd>
                  <dt className="text-muted">Target / origin</dt>
                  <dd className="font-mono">
                    {f.target_date} / {f.origin_date}
                  </dd>
                  <dt className="text-muted">Created</dt>
                  <dd className="font-mono">{fmtDateTimeIST(f.created_at)}</dd>
                </dl>
              </PopoverContent>
            </Popover>
          )}
          {f && (
            <button
              type="button"
              onClick={onCopy}
              className="hidden max-w-[15rem] items-center gap-1.5 rounded-md border border-border bg-elev2/70 px-2 py-1 font-mono text-2xs text-fg/90 hover:bg-elev3 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring min-[1760px]:inline-flex"
              aria-label={`Copy forecast ID ${f.forecast_id}`}
              title={`${f.forecast_id} · model ${f.model_version} · features ${f.feature_version}`}
            >
              <span className="truncate">{f.forecast_id}</span>
              {copied ? (
                <Check className="size-3 text-success" aria-hidden />
              ) : (
                <Copy className="size-3 text-muted" aria-hidden />
              )}
            </button>
          )}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2 xl:ml-auto">
          <DropdownMenu>
            <DropdownMenuTrigger
              className={cn(triggerCls, 'w-[7.5rem]')}
              aria-label={`Region: ${titleCase(p.region)}`}
              title="Region"
            >
              <span className="inline-flex min-w-0 items-center gap-1.5">
                <MapPin className="size-3.5 shrink-0 text-muted" aria-hidden />
                <span className="truncate">{titleCase(p.region)}</span>
              </span>
              <ChevronDown className="size-3.5 shrink-0 text-muted" aria-hidden />
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className={cn(scope, 'glass-strong')}>
              <DropdownMenuLabel>Region</DropdownMenuLabel>
              <DropdownMenuRadioGroup value={p.region} onValueChange={p.onRegion}>
                {p.regions.map((r) => (
                  <DropdownMenuRadioItem key={r} value={r}>
                    {titleCase(r)}
                  </DropdownMenuRadioItem>
                ))}
              </DropdownMenuRadioGroup>
            </DropdownMenuContent>
          </DropdownMenu>

          <Label htmlFor="studio-target-date" className="sr-only">
            Target date
          </Label>
          <Input
            id="studio-target-date"
            type="date"
            required
            value={p.targetDate}
            onChange={(e) => p.onTargetDate(e.target.value)}
            className="h-8 w-[8.75rem] font-mono"
            title="Target date"
          />

          <ToggleGroup
            type="single"
            value={String(p.horizon)}
            onValueChange={(v) => v && p.onHorizon(Number(v) as Horizon)}
            aria-label="Forecast horizon"
            className="h-8 p-0.5"
          >
            <ToggleGroupItem value="1" aria-label="T plus 1 day" className="h-6 px-2.5">
              T+1
            </ToggleGroupItem>
            <ToggleGroupItem value="2" aria-label="T plus 2 days" className="h-6 px-2.5">
              T+2
            </ToggleGroupItem>
          </ToggleGroup>

          <DropdownMenu>
            <DropdownMenuTrigger
              className={cn(triggerCls, 'w-[10.25rem]')}
              aria-label={`Model: ${modelLabel}`}
              title="Model"
            >
              <span className="truncate">{modelLabel}</span>
              <ChevronDown className="size-3.5 shrink-0 text-muted" aria-hidden />
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className={cn(scope, 'glass-strong min-w-[15rem]')}>
              <DropdownMenuLabel>
                Model{p.modelsFallback ? ' · offline list' : ''}
              </DropdownMenuLabel>
              <DropdownMenuRadioGroup value={p.model} onValueChange={p.onModel}>
                {p.models.map((m) => (
                  <DropdownMenuRadioItem key={m.key} value={m.key}>
                    <span className="flex w-full items-center justify-between gap-3">
                      <span>{m.label}</span>
                      <span className="font-mono text-2xs text-faint">
                        {m.quantiles ? 'P10–P90' : 'point'}
                      </span>
                    </span>
                  </DropdownMenuRadioItem>
                ))}
              </DropdownMenuRadioGroup>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {/* Actions */}
        <div className="ml-auto flex items-center gap-2 xl:ml-0">
          {p.dirty && f && (
            <span
              className="inline-flex items-center gap-1 text-2xs font-semibold text-warning"
              role="status"
              title="Parameters changed — run to update"
            >
              <span className="size-1.5 rounded-full bg-warning" aria-hidden />
              <span className="2xl:hidden">Changed</span>
              <span className="hidden 2xl:inline">Parameters changed</span>
            </span>
          )}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={p.onExport}
                disabled={!f || !f.blocks?.length}
                aria-label="Export the 96 blocks as CSV"
                className="h-8 px-2.5"
              >
                <Download aria-hidden />
                <span className="hidden 2xl:inline">Export CSV</span>
              </Button>
            </TooltipTrigger>
            <TooltipContent className={scope}>Export 96 blocks (CSV)</TooltipContent>
          </Tooltip>
          <Button
            type="submit"
            size="sm"
            disabled={p.running || !p.targetDate}
            className="h-8 min-w-[6.75rem]"
          >
            {p.running ? <Loader2 className="animate-spin" aria-hidden /> : <Play aria-hidden />}
            {p.running ? 'Running…' : 'Run forecast'}
          </Button>
        </div>
      </form>
    </header>
  );
}
