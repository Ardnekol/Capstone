import { Activity, Eye, History, Moon, Sun } from 'lucide-react';
import type { HealthStatus } from '@/types';

interface Props {
  isDark: boolean;
  onToggleTheme: () => void;
  onOpenHistory: () => void;
  health: HealthStatus | null;
}

export function Header({ isDark, onToggleTheme, onOpenHistory, health }: Props) {
  return (
    <header className="h-14 shrink-0 flex items-center justify-between px-5 border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-900/80 backdrop-blur-sm sticky top-0 z-40">
      {/* Brand */}
      <div className="flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg bg-violet-600 flex items-center justify-center shadow-lg shadow-violet-900/30">
          <Eye className="w-4 h-4 text-white" strokeWidth={2.5} />
        </div>
        <div className="leading-none">
          <span className="font-semibold text-slate-900 dark:text-slate-100 text-sm">Florence-2</span>
          <span className="text-slate-400 dark:text-slate-500 text-sm"> Playground</span>
        </div>
      </div>

      {/* Right controls */}
      <div className="flex items-center gap-1.5">
        {/* GPU badge */}
        {health && (
          <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-xs text-slate-500 dark:text-slate-400">
            <Activity className="w-3 h-3 text-emerald-500" />
            <span className="font-medium">{health.device.toUpperCase()}</span>
            {health.gpu_name && (
              <span className="text-slate-400 dark:text-slate-600 hidden lg:inline">
                · {health.gpu_name.replace('NVIDIA ', '').replace('GeForce ', '')}
              </span>
            )}
            {health.gpu_memory_used_gb != null && health.gpu_memory_total_gb != null && (
              <span className="text-slate-400 dark:text-slate-600 hidden lg:inline">
                {health.gpu_memory_used_gb.toFixed(1)}/{health.gpu_memory_total_gb.toFixed(0)} GB
              </span>
            )}
          </div>
        )}

        {/* History */}
        <button
          onClick={onOpenHistory}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
        >
          <History className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">History</span>
        </button>

        {/* Theme toggle */}
        <button
          onClick={onToggleTheme}
          className="p-1.5 rounded-lg text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          title={isDark ? 'Light mode' : 'Dark mode'}
        >
          {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>
      </div>
    </header>
  );
}
