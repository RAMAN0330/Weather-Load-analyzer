import { NavLink } from 'react-router-dom';
import {
  LineChart,
  CloudSun,
  Activity,
  Sliders,
  CheckCircle2,
  Settings as SettingsIcon,
  Zap,
  LogOut,
} from 'lucide-react';
import { cn } from '../../lib/utils';
import { Tooltip, TooltipContent, TooltipTrigger } from '../../components/ui/tooltip';
import { useAuthStore } from '../../features/auth/authStore';
import { useUi } from '../../store/ui';

interface NavItem {
  to: string;
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  badge?: string;
}

const NAV: NavItem[] = [
  { to: '/forecast', icon: LineChart, label: 'Forecast' },
  { to: '/weather', icon: CloudSun, label: 'Weather' },
  { to: '/monitor', icon: Activity, label: 'Monitor' },
  { to: '/simulator', icon: Sliders, label: 'Simulator' },
  { to: '/backtest', icon: CheckCircle2, label: 'Backtest', badge: 'NEW' },
];

export function Sidebar() {
  const collapsed = useUi((s) => s.sidebarCollapsed);
  const logout = useAuthStore((s: any) => s.logout);
  const user = useAuthStore((s: any) => s.user);

  return (
    <aside
      className={cn(
        'flex shrink-0 flex-col border-r border-border bg-elev1/60 backdrop-blur-md transition-all duration-200',
        collapsed ? 'w-16' : 'w-56'
      )}
    >
      {/* Brand */}
      <div className="flex items-center gap-2 px-4 py-5">
        <div className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-accent text-accent-foreground shadow-glow">
          <Zap className="h-4 w-4" />
        </div>
        {!collapsed && (
          <div className="overflow-hidden">
            <div className="font-display text-sm font-bold leading-tight text-fg">VidyutPragya</div>
            <div className="text-2xs uppercase tracking-wider text-faint">Forecast OS</div>
          </div>
        )}
      </div>

      {/* Nav */}
      <nav className="flex flex-1 flex-col gap-0.5 px-2">
        {NAV.map((item) => (
          <SidebarLink key={item.to} item={item} collapsed={collapsed} />
        ))}
      </nav>

      {/* Footer */}
      <div className="flex flex-col gap-0.5 border-t border-border p-2">
        <SidebarLink
          item={{ to: '/settings', icon: SettingsIcon, label: 'Settings' }}
          collapsed={collapsed}
        />
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              onClick={() => logout?.()}
              className={cn(
                'group flex h-9 items-center gap-3 rounded-lg px-3 text-sm font-medium text-muted transition-colors',
                'hover:bg-elev2 hover:text-fg',
                collapsed && 'justify-center px-0'
              )}
            >
              <LogOut className="h-4 w-4 shrink-0" />
              {!collapsed && <span className="truncate">Sign out</span>}
            </button>
          </TooltipTrigger>
          {collapsed && <TooltipContent side="right">Sign out · {user?.username || ''}</TooltipContent>}
        </Tooltip>
      </div>
    </aside>
  );
}

function SidebarLink({ item, collapsed }: { item: NavItem; collapsed: boolean }) {
  const Icon = item.icon;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <NavLink
          to={item.to}
          className={({ isActive }) =>
            cn(
              'group relative flex h-9 items-center gap-3 rounded-lg px-3 text-sm font-medium transition-all',
              isActive ? 'bg-elev2 text-fg' : 'text-muted hover:bg-elev2/60 hover:text-fg',
              collapsed && 'justify-center px-0'
            )
          }
        >
          {({ isActive }) => (
            <>
              {/* Active indicator bar */}
              <span
                className={cn(
                  'absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-full bg-accent transition-all',
                  isActive ? 'opacity-100' : 'opacity-0'
                )}
              />
              <Icon className="h-4 w-4 shrink-0" />
              {!collapsed && (
                <>
                  <span className="flex-1 truncate">{item.label}</span>
                  {item.badge && (
                    <span className="rounded-full bg-accent/20 px-1.5 py-0.5 text-2xs font-bold uppercase tracking-wider text-accent">
                      {item.badge}
                    </span>
                  )}
                </>
              )}
            </>
          )}
        </NavLink>
      </TooltipTrigger>
      {collapsed && <TooltipContent side="right">{item.label}</TooltipContent>}
    </Tooltip>
  );
}
