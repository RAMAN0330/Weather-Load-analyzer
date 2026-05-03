import * as React from 'react';
import * as ToggleGroupPrimitive from '@radix-ui/react-toggle-group';
import { cn } from '../../lib/utils';

export const ToggleGroup = React.forwardRef<
  React.ElementRef<typeof ToggleGroupPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof ToggleGroupPrimitive.Root>
>(({ className, ...props }, ref) => (
  <ToggleGroupPrimitive.Root
    ref={ref}
    className={cn(
      'inline-flex items-center gap-0.5 rounded-full border border-border bg-elev1 p-1',
      className
    )}
    {...props}
  />
));
ToggleGroup.displayName = ToggleGroupPrimitive.Root.displayName;

export const ToggleGroupItem = React.forwardRef<
  React.ElementRef<typeof ToggleGroupPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof ToggleGroupPrimitive.Item>
>(({ className, ...props }, ref) => (
  <ToggleGroupPrimitive.Item
    ref={ref}
    className={cn(
      'inline-flex h-7 items-center justify-center whitespace-nowrap rounded-full px-3 text-xs font-semibold uppercase tracking-wider text-muted transition-all duration-150',
      'data-[state=on]:bg-accent data-[state=on]:text-accent-foreground data-[state=on]:shadow-glow',
      'hover:text-fg disabled:pointer-events-none disabled:opacity-40',
      'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
      className
    )}
    {...props}
  />
));
ToggleGroupItem.displayName = ToggleGroupPrimitive.Item.displayName;
