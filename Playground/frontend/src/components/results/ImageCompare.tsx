import { AnimatePresence } from 'framer-motion';
import { useCallback, useEffect, useRef, useState } from 'react';
import { GripVertical } from 'lucide-react';
import { Lightbox } from './Lightbox';

interface Props {
  original: string;       // object URL of the uploaded file
  annotated: string;      // base64 PNG from inference
  /** for segmentation tasks — show an opacity slider instead of compare slider */
  showOpacityMode?: boolean;
}

export function ImageCompare({ original, annotated, showOpacityMode = false }: Props) {
  const [position, setPosition] = useState(50);
  const [opacity, setOpacity] = useState(0.75);
  const [dragging, setDragging] = useState(false);
  const [lightbox, setLightbox] = useState(false);
  const didDragRef = useRef(false);

  const [mode, setMode] = useState<'compare' | 'original' | 'annotated'>(
    showOpacityMode ? 'annotated' : 'compare'
  );
  const containerRef = useRef<HTMLDivElement>(null);

  const annotatedUrl = `data:image/png;base64,${annotated}`;

  // ── Compare drag logic ──────────────────────────────────────────────────────
  const updatePos = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const { left, width } = containerRef.current.getBoundingClientRect();
    setPosition(Math.min(Math.max(((clientX - left) / width) * 100, 1), 99));
  }, []);

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: MouseEvent | TouchEvent) => {
      didDragRef.current = true;
      const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
      updatePos(clientX);
    };
    const onUp = () => setDragging(false);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    window.addEventListener('touchmove', onMove, { passive: true });
    window.addEventListener('touchend', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      window.removeEventListener('touchmove', onMove);
      window.removeEventListener('touchend', onUp);
    };
  }, [dragging, updatePos]);

  // ── Mode toggle bar ─────────────────────────────────────────────────────────
  const tabs = showOpacityMode
    ? (['original', 'annotated'] as const)
    : (['compare', 'original', 'annotated'] as const);

  const lightboxSrc = mode === 'original' ? original : annotatedUrl;

  function handleContainerClick() {
    if (mode === 'compare') {
      // Only open lightbox for a plain click (no drag movement)
      if (!didDragRef.current) setLightbox(true);
    } else {
      setLightbox(true);
    }
  }

  return (
    <>
      <div className="flex flex-col gap-2">
        {/* Mode tabs */}
        <div className="flex items-center gap-1 self-start bg-slate-100 dark:bg-slate-800 rounded-lg p-0.5">
          {tabs.map((t) => (
            <button
              key={t}
              onClick={() => setMode(t)}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-all capitalize ${
                mode === t
                  ? 'bg-white dark:bg-slate-700 text-slate-800 dark:text-slate-100 shadow-sm'
                  : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
              }`}
            >
              {t}
            </button>
          ))}

          {/* Opacity slider for segmentation */}
          {showOpacityMode && mode === 'annotated' && (
            <div className="flex items-center gap-2 ml-2 pr-1">
              <span className="text-[10px] text-slate-400">Opacity</span>
              <input
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
                className="w-20 accent-violet-500 cursor-pointer"
              />
              <span className="text-[10px] text-slate-400 w-6 text-right">
                {Math.round(opacity * 100)}%
              </span>
            </div>
          )}
        </div>

        {/* Image area */}
        <div
          ref={containerRef}
          className="relative rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700 bg-slate-950/5 dark:bg-slate-950 select-none"
          style={{ cursor: mode === 'compare' ? 'col-resize' : 'zoom-in' }}
          onClick={handleContainerClick}
          onMouseDown={() => { didDragRef.current = false; }}
        >
          {/* ── Compare mode ──────────────────────────────────────────────── */}
          {mode === 'compare' && (
            <>
              <img
                src={annotatedUrl}
                alt="Annotated"
                className="w-full object-contain max-h-[420px] block"
                draggable={false}
              />
              <div
                className="absolute inset-0"
                style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
              >
                <img
                  src={original}
                  alt="Original"
                  className="w-full h-full object-contain"
                  draggable={false}
                />
              </div>

              {/* Divider */}
              <div
                className="absolute top-0 bottom-0 w-px bg-white/90 shadow-[0_0_6px_rgba(0,0,0,0.5)]"
                style={{ left: `${position}%` }}
              />

              {/* Drag handle */}
              <div
                className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-9 h-9 rounded-full bg-white shadow-xl flex items-center justify-center z-10 hover:scale-110 transition-transform"
                style={{ left: `${position}%` }}
                onMouseDown={(e) => { e.preventDefault(); e.stopPropagation(); didDragRef.current = false; setDragging(true); }}
                onTouchStart={(e) => { e.stopPropagation(); setDragging(true); }}
              >
                <GripVertical className="w-4 h-4 text-slate-600" />
              </div>

              <span className="absolute top-2 left-2 text-[10px] font-semibold text-white bg-black/50 px-2 py-0.5 rounded-full pointer-events-none">
                Original
              </span>
              <span className="absolute top-2 right-2 text-[10px] font-semibold text-white bg-violet-600/80 px-2 py-0.5 rounded-full pointer-events-none">
                Annotated
              </span>
            </>
          )}

          {/* ── Original only ─────────────────────────────────────────────── */}
          {mode === 'original' && (
            <img src={original} alt="Original" className="w-full object-contain max-h-[420px]" />
          )}

          {/* ── Annotated only ────────────────────────────────────────────── */}
          {mode === 'annotated' && (
            <div className="relative">
              <img src={original} alt="Original" className="w-full object-contain max-h-[420px]" />
              <img
                src={annotatedUrl}
                alt="Annotated overlay"
                className="absolute inset-0 w-full h-full object-contain"
                style={{ opacity: showOpacityMode ? opacity : 1 }}
              />
            </div>
          )}

          {/* Click-to-zoom hint */}
          <span className="absolute bottom-2 right-2 text-[10px] text-white/50 bg-black/30 px-1.5 py-0.5 rounded-full pointer-events-none select-none">
            click to enlarge
          </span>
        </div>
      </div>

      {/* ── Lightbox ──────────────────────────────────────────────────────── */}
      <AnimatePresence>
        {lightbox && (
          <Lightbox onClose={() => setLightbox(false)}>
            {mode === 'annotated' && showOpacityMode ? (
              // Show overlay with opacity in lightbox too
              <div className="relative inline-block rounded-xl overflow-hidden shadow-2xl">
                <img
                  src={original}
                  alt="Original"
                  className="block max-w-[88vw] max-h-[88vh]"
                  draggable={false}
                />
                <img
                  src={annotatedUrl}
                  alt="Overlay"
                  className="absolute inset-0 w-full h-full object-contain"
                  style={{ opacity }}
                />
              </div>
            ) : (
              <img
                src={lightboxSrc}
                alt="Enlarged"
                className="block max-w-[88vw] max-h-[88vh] rounded-xl shadow-2xl"
                draggable={false}
              />
            )}
          </Lightbox>
        )}
      </AnimatePresence>
    </>
  );
}
