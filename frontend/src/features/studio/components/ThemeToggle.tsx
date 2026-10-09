import { Monitor, Moon, Sun } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from '../../../components/ui/dropdown-menu';
import { Tooltip, TooltipContent, TooltipTrigger } from '../../../components/ui/tooltip';
import { cn } from '../../../lib/utils';
import { useResolvedTheme, useThemeMode, useThemeScopeClass, type ThemeMode } from '../theme';

const OPTIONS: { value: ThemeMode; label: string; Icon: typeof Sun }[] = [
  { value: 'light', label: 'Light', Icon: Sun },
  { value: 'dark', label: 'Dark', Icon: Moon },
  { value: 'system', label: 'System', Icon: Monitor },
];

/** Light / Dark / System selector (Radix menu: arrow keys, Enter/Space, Esc). */
export function ThemeToggle() {
  const [mode, setMode] = useThemeMode();
  const resolved = useResolvedTheme();
  const scope = useThemeScopeClass();
  const current = OPTIONS.find((o) => o.value === mode) ?? OPTIONS[2];
  const TriggerIcon = mode === 'system' ? Monitor : resolved === 'light' ? Sun : Moon;
  const label = `Theme: ${current.label}${mode === 'system' ? ` (${resolved})` : ''}`;

  return (
    <DropdownMenu>
      <Tooltip>
        <TooltipTrigger asChild>
          <DropdownMenuTrigger
            className="inline-flex size-8 items-center justify-center rounded-md border border-border bg-elev2/70 text-muted transition-colors hover:bg-elev3 hover:text-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-label={`${label}. Change theme`}
          >
            <TriggerIcon className="size-4" aria-hidden />
          </DropdownMenuTrigger>
        </TooltipTrigger>
        <TooltipContent className={scope}>{label}</TooltipContent>
      </Tooltip>
      <DropdownMenuContent align="end" className={cn(scope, 'glass-strong min-w-[10rem]')}>
        <DropdownMenuLabel>Appearance</DropdownMenuLabel>
        <DropdownMenuRadioGroup value={mode} onValueChange={(v) => setMode(v as ThemeMode)}>
          {OPTIONS.map(({ value, label: l, Icon }) => (
            <DropdownMenuRadioItem key={value} value={value} className="gap-2">
              <Icon className="size-3.5 text-muted" aria-hidden />
              {l}
              {value === 'system' && (
                <span className="ml-auto pl-3 font-mono text-2xs text-faint">{resolved}</span>
              )}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
