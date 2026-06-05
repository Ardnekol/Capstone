import { cn } from '@/lib/utils';
import type { TaskInfo } from '@/types';

interface Props {
  task: TaskInfo;
  selected: boolean;
  onClick: () => void;
}

export function TaskCard({ task, selected, onClick }: Props) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'w-full text-left px-3 py-2.5 rounded-lg border transition-all',
        selected
          ? 'border-violet-400/60 dark:border-violet-500/60 bg-violet-50 dark:bg-violet-500/10'
          : 'border-slate-200 dark:border-slate-700/60 bg-transparent hover:border-slate-300 dark:hover:border-slate-600 hover:bg-slate-50 dark:hover:bg-slate-800/60',
      )}
    >
      <p
        className={cn(
          'text-sm font-medium leading-snug',
          selected
            ? 'text-violet-700 dark:text-violet-300'
            : 'text-slate-700 dark:text-slate-200',
        )}
      >
        {task.name}
      </p>
      <p className="text-xs text-slate-400 dark:text-slate-500 mt-0.5 line-clamp-2 leading-relaxed">
        {task.description}
      </p>
    </button>
  );
}
