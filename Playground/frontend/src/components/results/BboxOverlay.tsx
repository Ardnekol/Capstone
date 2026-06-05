import { AnimatePresence, motion } from 'framer-motion';
import { useRef, useState } from 'react';
import { Lightbox } from './Lightbox';

interface Props {
  imageUrl: string;
  bboxes: number[][];
  labels: string[];
}

function goldenPalette(n: number): string[] {
  return Array.from({ length: Math.max(n, 1) }, (_, i) => {
    const hue = (i * 137.508) % 360;
    const h = hue / 60;
    const c = 200;
    const x = Math.round(c * (1 - Math.abs((h % 2) - 1)));
    const s = Math.floor(h);
    let r = 0, g = 0, b = 0;
    if      (s === 0) { r = c; g = x; b = 0; }
    else if (s === 1) { r = x; g = c; b = 0; }
    else if (s === 2) { r = 0; g = c; b = x; }
    else if (s === 3) { r = 0; g = x; b = c; }
    else if (s === 4) { r = x; g = 0; b = c; }
    else              { r = c; g = 0; b = x; }
    return `${r},${g},${b}`;
  });
}

// Extracted so it renders identically in both the card and the lightbox
function BboxSvgLayer({
  natural, bboxes, labels, colors, animate = true,
}: {
  natural: { w: number; h: number };
  bboxes: number[][];
  labels: string[];
  colors: string[];
  animate?: boolean;
}) {
  return (
    <svg
      viewBox={`0 0 ${natural.w} ${natural.h}`}
      className="absolute inset-0 w-full h-full"
      style={{ pointerEvents: 'none' }}
    >
      {bboxes.map((bbox, i) => {
        const [x1, y1, x2, y2] = bbox.map(Math.round);
        const w = x2 - x1;
        const h = y2 - y1;
        const perimeter = 2 * (w + h);
        const rgb = colors[i % colors.length];
        const delay = animate ? i * 0.07 : 0;
        const label = labels[i] ?? '';

        return (
          <g key={i}>
            {/* Semi-transparent fill */}
            <motion.rect
              x={x1} y={y1} width={w} height={h}
              fill={`rgba(${rgb},0.12)`}
              stroke="none"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.25, delay: delay + 0.4 }}
            />

            {/* Border — non-scaling-stroke keeps it 3 screen-px regardless of image resolution */}
            <motion.rect
              x={x1} y={y1} width={w} height={h}
              fill="none"
              stroke={`rgb(${rgb})`}
              strokeWidth={3}
              style={{ vectorEffect: 'non-scaling-stroke' } as React.CSSProperties}
              strokeDasharray={perimeter}
              initial={{ strokeDashoffset: perimeter, opacity: 1 }}
              animate={{ strokeDashoffset: 0 }}
              transition={{ duration: 0.45, delay, ease: 'easeInOut' }}
            />

            {/* Label pill */}
            {label && (
              <motion.g
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.2, delay: delay + 0.5 }}
              >
                <rect
                  x={x1}
                  y={Math.max(0, y1 - 20)}
                  width={Math.min(label.length * 7 + 12, natural.w - x1)}
                  height={20}
                  fill={`rgba(${rgb},0.88)`}
                  rx={4}
                />
                <text
                  x={x1 + 6}
                  y={Math.max(0, y1 - 20) + 14}
                  fontSize={12}
                  fontFamily="sans-serif"
                  fontWeight="700"
                  fill="white"
                >
                  {label.slice(0, 32)}
                </text>
              </motion.g>
            )}
          </g>
        );
      })}
    </svg>
  );
}

export function BboxOverlay({ imageUrl, bboxes, labels }: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const [lightbox, setLightbox] = useState(false);

  const colors = goldenPalette(bboxes.length);

  return (
    <>
      {/* ── Card view ─────────────────────────────────────────────────────── */}
      <div
        className="relative rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700 bg-slate-950/5 dark:bg-slate-950 inline-block w-full cursor-zoom-in"
        onClick={() => setLightbox(true)}
        title="Click to enlarge"
      >
        <img
          ref={imgRef}
          src={imageUrl}
          alt="Input"
          className="w-full object-contain max-h-[420px] block"
          draggable={false}
          onLoad={() => {
            if (imgRef.current)
              setNatural({ w: imgRef.current.naturalWidth, h: imgRef.current.naturalHeight });
          }}
        />
        {natural && (
          <BboxSvgLayer natural={natural} bboxes={bboxes} labels={labels} colors={colors} />
        )}
      </div>

      {/* ── Lightbox ──────────────────────────────────────────────────────── */}
      <AnimatePresence>
        {lightbox && natural && (
          <Lightbox onClose={() => setLightbox(false)}>
            <div className="relative inline-block rounded-xl overflow-hidden shadow-2xl">
              <img
                src={imageUrl}
                alt="Enlarged"
                className="block max-w-[88vw] max-h-[88vh]"
                draggable={false}
              />
              <BboxSvgLayer
                natural={natural}
                bboxes={bboxes}
                labels={labels}
                colors={colors}
                animate={false}
              />
            </div>
          </Lightbox>
        )}
      </AnimatePresence>
    </>
  );
}
