import { FileText, Layers, Link2, MapPin, ScanSearch, Type, type LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { TaskGroupId } from '@/types';

interface GroupMeta {
  label: string;
  icon: LucideIcon;
  activeText: string;
  activeBg: string;
  dot: string;
}

const GROUPS: Record<TaskGroupId, GroupMeta> = {
  caption: {
    label: 'Captioning',
    icon: FileText,
    activeText: 'text-violet-600 dark:text-violet-400',
    activeBg: 'bg-violet-50 dark:bg-violet-500/10',
    dot: 'bg-violet-500',
  },
  detection: {
    label: 'Detection',
    icon: ScanSearch,
    activeText: 'text-blue-600 dark:text-blue-400',
    activeBg: 'bg-blue-50 dark:bg-blue-500/10',
    dot: 'bg-blue-500',
  },
  grounding: {
    label: 'Grounding',
    icon: MapPin,
    activeText: 'text-emerald-600 dark:text-emerald-400',
    activeBg: 'bg-emerald-50 dark:bg-emerald-500/10',
    dot: 'bg-emerald-500',
  },
  segmentation: {
    label: 'Segmentation',
    icon: Layers,
    activeText: 'text-orange-600 dark:text-orange-400',
    activeBg: 'bg-orange-50 dark:bg-orange-500/10',
    dot: 'bg-orange-500',
  },
  ocr: {
    label: 'OCR',
    icon: Type,
    activeText: 'text-yellow-600 dark:text-yellow-400',
    activeBg: 'bg-yellow-50 dark:bg-yellow-500/10',
    dot: 'bg-yellow-500',
  },
  cascaded: {
    label: 'Cascaded',
    icon: Link2,
    activeText: 'text-pink-600 dark:text-pink-400',
    activeBg: 'bg-pink-50 dark:bg-pink-500/10',
    dot: 'bg-pink-500',
  },
};

interface Props {
  selectedGroup: TaskGroupId;
  onSelectGroup: (g: TaskGroupId) => void;
}

export function Sidebar({ selectedGroup, onSelectGroup }: Props) {
  return (
    <aside className="w-48 shrink-0 border-r border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 flex flex-col py-4 px-2 gap-0.5 overflow-y-auto">
      <p className="text-[10px] font-semibold tracking-widest text-slate-400 dark:text-slate-600 uppercase px-2.5 mb-2">
        Task Groups
      </p>

      {(Object.entries(GROUPS) as [TaskGroupId, GroupMeta][]).map(([id, meta]) => {
        const Icon = meta.icon;
        const active = selectedGroup === id;
        return (
          <button
            key={id}
            onClick={() => onSelectGroup(id)}
            className={cn(
              'group flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-sm font-medium transition-all text-left',
              active
                ? cn(meta.activeBg, meta.activeText)
                : 'text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-100',
            )}
          >
            <Icon
              className={cn(
                'w-4 h-4 shrink-0 transition-colors',
                active ? meta.activeText : 'text-slate-400 dark:text-slate-600 group-hover:text-current',
              )}
            />
            <span>{meta.label}</span>
            {active && <span className={cn('ml-auto w-1.5 h-1.5 rounded-full', meta.dot)} />}
          </button>
        );
      })}
    </aside>
  );
}
