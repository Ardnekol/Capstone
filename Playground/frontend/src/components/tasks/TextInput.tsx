import type { TaskInfo } from '@/types';

interface Props {
  taskInfo: TaskInfo;
  value: string;
  onChange: (v: string) => void;
}

export function TextInput({ taskInfo, value, onChange }: Props) {
  const isRegion = taskInfo.needs_region_input;

  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wide flex items-center gap-1">
        {isRegion ? 'Region Coordinates' : 'Text Prompt'}
        <span className="text-rose-500 ml-0.5 normal-case font-normal tracking-normal">required</span>
      </label>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={taskInfo.text_input_placeholder ?? ''}
        className="w-full px-3 py-2 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-sm text-slate-800 dark:text-slate-200 placeholder:text-slate-300 dark:placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-violet-500/40 focus:border-violet-400 transition-colors font-mono"
      />
      {isRegion && (
        <p className="text-xs text-slate-400 dark:text-slate-600">
          Pixel coordinates: x1, y1 (top-left) and x2, y2 (bottom-right)
        </p>
      )}
    </div>
  );
}
