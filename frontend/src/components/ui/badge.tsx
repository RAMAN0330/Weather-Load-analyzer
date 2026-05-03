import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '../../lib/utils';

const badgeVariants = cva(
  'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-2xs font-semibold uppercase tracking-wider transition-colors',
  {
    variants: {
      variant: {
        default: 'border-border bg-elev2 text-fg',
        accent: 'border-accent/40 bg-accent/15 text-accent',
        success: 'border-success/40 bg-success/15 text-success',
        warning: 'border-warning/40 bg-warning/15 text-warning',
        danger: 'border-danger/40 bg-danger/15 text-danger',
        info: 'border-info/40 bg-info/15 text-info',
        outline: 'border-border bg-transparent text-muted',
      },
    },
    defaultVariants: { variant: 'default' },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
