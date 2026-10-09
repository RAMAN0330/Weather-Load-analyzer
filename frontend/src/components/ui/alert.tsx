import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '../../lib/utils';

const alertVariants = cva(
  'relative w-full rounded-lg border px-4 py-3 text-sm [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-3.5 [&>svg]:size-4 [&>svg+div]:translate-y-[-1px] [&:has(>svg)]:pl-11',
  {
    variants: {
      variant: {
        default: 'border-border bg-elev1 text-fg [&>svg]:text-muted',
        info: 'border-info/40 bg-info/10 text-fg [&>svg]:text-info',
        warning: 'border-warning/45 bg-warning/10 text-fg [&>svg]:text-warning',
        destructive: 'border-danger/50 bg-danger/10 text-fg [&>svg]:text-danger',
        success: 'border-success/40 bg-success/10 text-fg [&>svg]:text-success',
      },
    },
    defaultVariants: { variant: 'default' },
  }
);

export const Alert = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof alertVariants>
>(({ className, variant, ...props }, ref) => (
  <div ref={ref} role="alert" className={cn(alertVariants({ variant }), className)} {...props} />
));
Alert.displayName = 'Alert';

export const AlertTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h5
    ref={ref}
    className={cn('mb-1 font-semibold leading-none tracking-tight', className)}
    {...props}
  />
));
AlertTitle.displayName = 'AlertTitle';

export const AlertDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn('text-sm text-fg/85 [&_p]:leading-relaxed', className)} {...props} />
));
AlertDescription.displayName = 'AlertDescription';

export { alertVariants };
