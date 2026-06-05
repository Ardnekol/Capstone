import { motion } from 'framer-motion';
import { ZoomIn } from 'lucide-react';
import { useState } from 'react';

interface Props {
  src: string; // base64 PNG (no data-URL prefix)
}

export function AnnotatedImage({ src }: Props) {
  const [lightbox, setLightbox] = useState(false);
  const dataUrl = `data:image/png;base64,${src}`;

  return (
    <>
      <motion.div
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.2 }}
        className="relative group rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700 bg-slate-950/5 dark:bg-slate-950 cursor-zoom-in"
        onClick={() => setLightbox(true)}
      >
        <img
          src={dataUrl}
          alt="Annotated result"
          className="w-full object-contain max-h-[420px]"
        />
        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center">
          <ZoomIn className="w-6 h-6 text-white opacity-0 group-hover:opacity-80 transition-opacity drop-shadow" />
        </div>
      </motion.div>

      {/* Lightbox */}
      {lightbox && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4 cursor-zoom-out"
          onClick={() => setLightbox(false)}
        >
          <img
            src={dataUrl}
            alt="Full size annotated result"
            className="max-w-full max-h-full object-contain rounded-lg"
          />
        </div>
      )}
    </>
  );
}
