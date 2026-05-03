import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { useUrlState } from '../../hooks/useUrlState';

export function AppShell() {
  // Keep URL <-> store in sync at the shell level so every page benefits.
  useUrlState();

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar />
        <main className="flex min-h-0 flex-1 flex-col overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
