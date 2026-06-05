import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useState } from 'react';

interface Props {
  loading: boolean;
}

export function ProgressBar({ loading }: Props) {
  const [progress, setProgress] = useState(0);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    let timers: ReturnType<typeof setTimeout>[] = [];

    if (loading) {
      setVisible(true);
      setProgress(0);
      // Fake incremental progress: fast at first, then slows down
      timers = [
        setTimeout(() => setProgress(25), 80),
        setTimeout(() => setProgress(50), 500),
        setTimeout(() => setProgress(72), 1500),
        setTimeout(() => setProgress(85), 4000),
        setTimeout(() => setProgress(92), 9000),
      ];
    } else {
      setProgress(100);
      const t = setTimeout(() => {
        setVisible(false);
        setProgress(0);
      }, 400);
      timers = [t];
    }

    return () => timers.forEach(clearTimeout);
  }, [loading]);

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          key="progress-bar"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="fixed top-0 left-0 right-0 h-0.5 z-50 bg-slate-200 dark:bg-slate-800"
        >
          <motion.div
            className="h-full bg-gradient-to-r from-violet-500 via-violet-400 to-violet-500 rounded-r-full"
            animate={{ width: `${progress}%` }}
            transition={{ duration: progress === 100 ? 0.2 : 0.6, ease: 'easeOut' }}
          />
          {/* Shimmer glow at the tip */}
          <motion.div
            className="absolute top-0 bottom-0 w-12 bg-gradient-to-r from-transparent to-white/40 dark:to-white/20 rounded-full"
            animate={{ left: `calc(${progress}% - 3rem)` }}
            transition={{ duration: progress === 100 ? 0.2 : 0.6, ease: 'easeOut' }}
          />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
