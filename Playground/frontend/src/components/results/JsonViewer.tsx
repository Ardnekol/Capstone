import { useState } from 'react';
import { ChevronDown, ChevronRight, Code2 } from 'lucide-react';

interface Props {
  data: unknown;
  defaultOpen?: boolean;
}

export function JsonViewer({ data, defaultOpen = false }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  const json = JSON.stringify(data, null, 2);

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-3.5 py-2.5 bg-slate-50 dark:bg-slate-800/60 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
      >
        {open ? (
          <ChevronDown className="w-3.5 h-3.5 text-slate-400 shrink-0" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 text-slate-400 shrink-0" />
        )}
        <Code2 className="w-3.5 h-3.5 text-slate-400 shrink-0" />
        <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">
          Raw JSON Output
        </span>
        <span className="ml-auto text-xs text-slate-300 dark:text-slate-600 font-mono">
          {json.length.toLocaleString()} chars
        </span>
      </button>

      {open && (
        <pre className="p-4 text-xs font-mono text-slate-600 dark:text-slate-300 overflow-auto max-h-72 scrollbar-thin bg-white dark:bg-slate-950 leading-relaxed whitespace-pre-wrap break-words">
          {json}
        </pre>
      )}
    </div>
  );
}
