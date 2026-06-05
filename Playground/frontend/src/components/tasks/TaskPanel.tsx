import { Loader2, Play } from 'lucide-react';
import { TaskCard } from './TaskCard';
import { TextInput } from './TextInput';
import { cn } from '@/lib/utils';
import type { TaskInfo } from '@/types';

interface Props {
  tasks: TaskInfo[];
  selectedTask: string;
  onSelectTask: (key: string) => void;
  taskInfo: TaskInfo | undefined;
  textInput: string;
  onTextInput: (v: string) => void;
  models: string[];
  modelId: string;
  onModelChange: (m: string) => void;
  onRun: () => void;
  loading: boolean;
  canRun: boolean;
}

export function TaskPanel({
  tasks,
  selectedTask,
  onSelectTask,
  taskInfo,
  textInput,
  onTextInput,
  models,
  modelId,
  onModelChange,
  onRun,
  loading,
  canRun,
}: Props) {
  const needsInput = taskInfo?.needs_text_input || taskInfo?.needs_region_input;
  const inputReady = !needsInput || textInput.trim().length > 0;
  const runnable = canRun && inputReady && !loading;

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 flex flex-col overflow-hidden">
      {/* Task selection */}
      <div className="p-3 border-b border-slate-100 dark:border-slate-800">
        <p className="text-[10px] font-semibold tracking-widest text-slate-400 dark:text-slate-600 uppercase mb-2 px-0.5">
          Select Task
        </p>
        <div className="flex flex-col gap-1.5 max-h-56 overflow-y-auto scrollbar-thin pr-0.5">
          {tasks.length === 0 ? (
            <div className="py-4 text-center text-xs text-slate-400">Loading tasks…</div>
          ) : (
            tasks.map((t) => (
              <TaskCard
                key={t.key}
                task={t}
                selected={selectedTask === t.key}
                onClick={() => onSelectTask(t.key)}
              />
            ))
          )}
        </div>
      </div>

      {/* Text / region input */}
      {taskInfo && needsInput && (
        <div className="px-3 py-3 border-b border-slate-100 dark:border-slate-800">
          <TextInput taskInfo={taskInfo} value={textInput} onChange={onTextInput} />
        </div>
      )}

      {/* Model selector + Run */}
      <div className="p-3 flex flex-col gap-3">
        {models.length > 0 && (
          <div className="flex flex-col gap-1.5">
            <label className="text-[10px] font-semibold tracking-widest text-slate-400 dark:text-slate-600 uppercase">
              Model
            </label>
            <select
              value={modelId}
              onChange={(e) => onModelChange(e.target.value)}
              disabled={loading}
              className="w-full px-3 py-2 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-sm text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-violet-500/40 focus:border-violet-400 transition-colors disabled:opacity-50"
            >
              {models.map((m) => (
                <option key={m} value={m}>
                  {m.replace('microsoft/', '')}
                </option>
              ))}
            </select>
          </div>
        )}

        <button
          onClick={onRun}
          disabled={!runnable}
          className={cn(
            'w-full flex items-center justify-center gap-2 py-2.5 rounded-lg font-semibold text-sm transition-all select-none',
            runnable
              ? 'bg-violet-600 hover:bg-violet-500 active:bg-violet-700 text-white shadow-md shadow-violet-900/20 hover:shadow-violet-900/30'
              : 'bg-slate-100 dark:bg-slate-800 text-slate-300 dark:text-slate-600 cursor-not-allowed',
          )}
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Running…
            </>
          ) : (
            <>
              <Play className="w-3.5 h-3.5" fill="currentColor" />
              Run Inference
              <kbd className="ml-auto text-[10px] opacity-40 font-mono bg-white/10 px-1 py-0.5 rounded">
                ⌘↵
              </kbd>
            </>
          )}
        </button>

        {!canRun && !loading && (
          <p className="text-center text-xs text-slate-400 dark:text-slate-600">
            Upload an image to enable inference
          </p>
        )}
      </div>
    </div>
  );
}
