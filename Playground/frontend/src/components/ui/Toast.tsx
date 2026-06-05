import { AnimatePresence, motion } from 'framer-motion';
import { AlertCircle, CheckCircle2, Info, X } from 'lucide-react';
import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';

// ── Types ─────────────────────────────────────────────────────────────────────

export type ToastType = 'success' | 'error' | 'info';

interface ToastItem {
  id: string;
  type: ToastType;
  title: string;
  message?: string;
  duration: number;
}

interface ToastContextValue {
  toast: (opts: { type: ToastType; title: string; message?: string; duration?: number }) => void;
}

// ── Context ───────────────────────────────────────────────────────────────────

const ToastContext = createContext<ToastContextValue | null>(null);

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used inside <ToastProvider>');
  return ctx;
}

// ── Single Toast item ─────────────────────────────────────────────────────────

const ICON = {
  success: <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />,
  error:   <AlertCircle  className="w-4 h-4 text-rose-400    shrink-0 mt-0.5" />,
  info:    <Info         className="w-4 h-4 text-violet-400  shrink-0 mt-0.5" />,
};

const BAR_COLOR = {
  success: 'bg-emerald-500',
  error:   'bg-rose-500',
  info:    'bg-violet-500',
};

function ToastCard({
  item,
  onDismiss,
}: {
  item: ToastItem;
  onDismiss: (id: string) => void;
}) {
  const [paused, setPaused] = useState(false);
  const elapsed = useRef(0);
  const lastTick = useRef(Date.now());

  // Auto-dismiss timer — pauses on hover
  useEffect(() => {
    const tick = () => {
      if (!paused) {
        elapsed.current += Date.now() - lastTick.current;
        if (elapsed.current >= item.duration) onDismiss(item.id);
      }
      lastTick.current = Date.now();
    };
    const id = setInterval(tick, 50);
    return () => clearInterval(id);
  }, [paused, item.id, item.duration, onDismiss]);

  const progress = Math.min((elapsed.current / item.duration) * 100, 100);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: 40, scale: 0.95 }}
      animate={{ opacity: 1, x: 0,  scale: 1    }}
      exit={{    opacity: 0, x: 40, scale: 0.95, transition: { duration: 0.15 } }}
      transition={{ type: 'spring', damping: 22, stiffness: 260 }}
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => { setPaused(false); lastTick.current = Date.now(); }}
      className="w-80 rounded-xl border border-slate-700/60 bg-slate-900 shadow-2xl shadow-black/40 overflow-hidden"
    >
      {/* Body */}
      <div className="flex items-start gap-3 px-4 py-3.5">
        {ICON[item.type]}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-slate-100 leading-snug">{item.title}</p>
          {item.message && (
            <p className="text-xs text-slate-400 mt-0.5 leading-relaxed">{item.message}</p>
          )}
        </div>
        <button
          onClick={() => onDismiss(item.id)}
          className="p-1 rounded-md text-slate-500 hover:text-slate-200 hover:bg-slate-800 transition-colors shrink-0"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Progress bar */}
      <div className="h-0.5 bg-slate-800">
        <motion.div
          className={`h-full ${BAR_COLOR[item.type]}`}
          animate={{ width: paused ? undefined : `${100 - progress}%` }}
          transition={{ duration: 0.05, ease: 'linear' }}
          style={{ width: `${100 - progress}%` }}
        />
      </div>
    </motion.div>
  );
}

// ── Container ─────────────────────────────────────────────────────────────────

function ToastContainer({ toasts, onDismiss }: { toasts: ToastItem[]; onDismiss: (id: string) => void }) {
  return (
    <div className="fixed bottom-5 right-5 z-[100] flex flex-col gap-2 items-end pointer-events-none">
      <AnimatePresence mode="popLayout">
        {toasts.map((t) => (
          <div key={t.id} className="pointer-events-auto">
            <ToastCard item={t} onDismiss={onDismiss} />
          </div>
        ))}
      </AnimatePresence>
    </div>
  );
}

// ── Provider ──────────────────────────────────────────────────────────────────

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const toast = useCallback(
    ({ type, title, message, duration = 3500 }: {
      type: ToastType;
      title: string;
      message?: string;
      duration?: number;
    }) => {
      const id = crypto.randomUUID();
      setToasts((prev) => [...prev.slice(-4), { id, type, title, message, duration }]);
    },
    [],
  );

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      <ToastContainer toasts={toasts} onDismiss={dismiss} />
    </ToastContext.Provider>
  );
}
