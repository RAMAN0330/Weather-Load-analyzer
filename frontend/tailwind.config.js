/** @type {import('tailwindcss').Config} */
import animate from 'tailwindcss-animate';

export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  darkMode: 'class',
  theme: {
    container: {
      center: true,
      padding: '1rem',
    },
    extend: {
      colors: {
        // Surfaces
        base: 'hsl(var(--bg-base) / <alpha-value>)',
        elev1: 'hsl(var(--bg-elev-1) / <alpha-value>)',
        elev2: 'hsl(var(--bg-elev-2) / <alpha-value>)',
        elev3: 'hsl(var(--bg-elev-3) / <alpha-value>)',

        border: 'hsl(var(--border) / <alpha-value>)',
        ring: 'hsl(var(--ring) / <alpha-value>)',

        // Text
        fg: 'hsl(var(--fg) / <alpha-value>)',
        muted: 'hsl(var(--fg-muted) / <alpha-value>)',
        faint: 'hsl(var(--fg-faint) / <alpha-value>)',

        // Brand / semantic
        accent: {
          DEFAULT: 'hsl(var(--accent) / <alpha-value>)',
          foreground: 'hsl(var(--accent-fg) / <alpha-value>)',
        },
        accent2: {
          DEFAULT: 'hsl(var(--accent-2) / <alpha-value>)',
          foreground: 'hsl(var(--accent-2-fg) / <alpha-value>)',
        },
        success: 'hsl(var(--success) / <alpha-value>)',
        warning: 'hsl(var(--warning) / <alpha-value>)',
        danger: 'hsl(var(--danger) / <alpha-value>)',
        info: 'hsl(var(--info) / <alpha-value>)',

        // shadcn aliases
        background: 'hsl(var(--bg-base) / <alpha-value>)',
        foreground: 'hsl(var(--fg) / <alpha-value>)',
        card: {
          DEFAULT: 'hsl(var(--bg-elev-1) / <alpha-value>)',
          foreground: 'hsl(var(--fg) / <alpha-value>)',
        },
        popover: {
          DEFAULT: 'hsl(var(--bg-elev-2) / <alpha-value>)',
          foreground: 'hsl(var(--fg) / <alpha-value>)',
        },
        primary: {
          DEFAULT: 'hsl(var(--accent) / <alpha-value>)',
          foreground: 'hsl(var(--accent-fg) / <alpha-value>)',
        },
        secondary: {
          DEFAULT: 'hsl(var(--bg-elev-2) / <alpha-value>)',
          foreground: 'hsl(var(--fg) / <alpha-value>)',
        },
        destructive: {
          DEFAULT: 'hsl(var(--danger) / <alpha-value>)',
          foreground: 'hsl(0 0% 100% / <alpha-value>)',
        },
        input: 'hsl(var(--bg-elev-2) / <alpha-value>)',
      },
      borderRadius: {
        sm: '4px',
        md: '8px',
        lg: '12px',
        xl: '16px',
        '2xl': '20px',
      },
      fontFamily: {
        sans: ['Manrope', 'Inter', 'system-ui', 'sans-serif'],
        display: ['Space Grotesk', 'Manrope', 'sans-serif'],
        mono: ['IBM Plex Mono', 'JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      fontSize: {
        '2xs': ['11px', { lineHeight: '14px' }],
      },
      keyframes: {
        'accordion-down': {
          from: { height: '0' },
          to: { height: 'var(--radix-accordion-content-height)' },
        },
        'accordion-up': {
          from: { height: 'var(--radix-accordion-content-height)' },
          to: { height: '0' },
        },
        'fade-in': {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        'slide-in-up': {
          from: { opacity: '0', transform: 'translateY(8px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
        'fade-in': 'fade-in 200ms ease-out',
        'slide-in-up': 'slide-in-up 240ms ease-out',
      },
      boxShadow: {
        glow: '0 0 0 1px hsl(var(--accent) / 0.3), 0 8px 30px hsl(var(--accent) / 0.18)',
        elev1: '0 1px 0 0 hsl(0 0% 100% / 0.04) inset, 0 8px 30px hsl(0 0% 0% / 0.45)',
      },
    },
  },
  plugins: [animate],
};
