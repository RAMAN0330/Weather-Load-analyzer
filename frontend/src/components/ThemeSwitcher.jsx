import React from 'react';
import { TooltipProvider } from './ui/tooltip';
import { ThemeToggle } from '../features/studio/components/ThemeToggle';
import { useThemeScopeClass } from '../features/studio/theme';

/** Light / Dark (One Dark Pro) / System switcher usable on any page (login, landing, app). */
export default function ThemeSwitcher({ className = '' }) {
  const scope = useThemeScopeClass();
  return (
    <span className={`${scope} ${className}`.trim()}>
      <TooltipProvider delayDuration={300}>
        <ThemeToggle />
      </TooltipProvider>
    </span>
  );
}
