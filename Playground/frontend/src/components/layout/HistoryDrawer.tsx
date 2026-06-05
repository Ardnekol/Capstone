import { AnimatePresence, motion } from 'framer-motion';
import { Clock, Trash2, X } from 'lucide-react';
import { formatTimestamp } from '@/lib/utils';
import type { HistoryItem } from '@/types';

interface Props {
  open: boolean;
  onClose: () => void;
  history: HistoryItem[];
  onClear: () => void;
  onSelect: (item: HistoryItem) => void;
}

const GROUP_COLORS: Record<string, string> = {
  caption: 'bg-violet-500/20 text-violet-400',
  detection: 'bg-blue-500/20 text-blue-400',
  grounding: 'bg-emerald-500/20 text-emerald-400',
  segmentation: 'bg-orange-500/20 text-orange-400',
  ocr: 'bg-yellow-500/20 text-yellow-400',
  cascaded: 'bg-pink-500/20 text-pink-400',
};

export function HistoryDrawer({ open, onClose, history, onClear, onSelect }: Props) {
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            key="backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/40 dark:bg-black/60 z-40 backdrop-blur-sm"
          />

          <motion.aside
            key="drawer"
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 28, stiffness: 220 }}
            className="fixed right-0 top-0 h-full w-80 bg-white dark:bg-slate-900 border-l border-slate-200 dark:border-slate-800 z-50 flex flex-col shadow-2xl"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3.5 border-b border-slate-200 dark:border-slate-800">
              <div className="flex items-center gap-2">
                <Clock className="w-3.5 h-3.5 text-slate-400" />
                <span className="font-semibold text-slate-800 dark:text-slate-200 text-sm">
                  Run History
                </span>
                {history.length > 0 && (
                  <span className="text-xs text-slate-400 bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded-full">
                    {history.length}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1">
                {history.length > 0 && (
                  <button
                    onClick={onClear}
                    className="p-1.5 rounded-lg text-slate-400 hover:text-rose-500 hover:bg-rose-50 dark:hover:bg-rose-500/10 transition-colors"
                    title="Clear all history"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
                <button
                  onClick={onClose}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* Items */}
            <div className="flex-1 overflow-y-auto scrollbar-thin p-2">
              {history.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-48 text-center">
                  <Clock className="w-10 h-10 text-slate-200 dark:text-slate-700 mb-3" />
                  <p className="text-sm text-slate-400">No runs yet</p>
                  <p className="text-xs text-slate-300 dark:text-slate-600 mt-1">
                    Your inference history will appear here
                  </p>
                </div>
              ) : (
                <div className="flex flex-col gap-1">
                  {history.map((item) => (
                    <motion.button
                      key={item.id}
                      initial={{ opacity: 0, x: 10 }}
                      animate={{ opacity: 1, x: 0 }}
                      onClick={() => onSelect(item)}
                      className="flex items-center gap-3 p-2.5 rounded-xl hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors text-left w-full group"
                    >
                      <img
                        src={item.thumbnail}
                        alt=""
                        className="w-12 h-12 rounded-lg object-cover border border-slate-200 dark:border-slate-700 shrink-0"
                      />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-slate-700 dark:text-slate-200 truncate">
                          {item.task_name}
                        </p>
                        <div className="flex items-center gap-1.5 mt-0.5">
                          <span
                            className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${
                              GROUP_COLORS[item.group] ?? 'bg-slate-200 text-slate-500'
                            }`}
                          >
                            {item.group}
                          </span>
                          <span className="text-xs text-slate-400">
                            {formatTimestamp(item.timestamp)}
                          </span>
                        </div>
                      </div>
                    </motion.button>
                  ))}
                </div>
              )}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
