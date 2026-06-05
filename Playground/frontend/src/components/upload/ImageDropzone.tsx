import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { AnimatePresence, motion } from 'framer-motion';
import { ImageIcon, Upload, X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Props {
  imageUrl: string | null;
  onFile: (f: File) => void;
  onClear: () => void;
}

export function ImageDropzone({ imageUrl, onFile, onClear }: Props) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted[0]) onFile(accepted[0]);
    },
    [onFile],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'] },
    maxFiles: 1,
  });

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 overflow-hidden">
      {/* Title bar */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-slate-100 dark:border-slate-800">
        <span className="text-xs font-semibold tracking-wide text-slate-500 dark:text-slate-400 uppercase flex items-center gap-1.5">
          <ImageIcon className="w-3.5 h-3.5" />
          Input Image
        </span>
        {imageUrl && (
          <button
            onClick={onClear}
            className="p-1 rounded-md text-slate-400 hover:text-rose-500 hover:bg-rose-50 dark:hover:bg-rose-500/10 transition-colors"
            title="Remove image"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      <AnimatePresence mode="wait">
        {imageUrl ? (
          <motion.div
            key="preview"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="relative group cursor-pointer"
            {...getRootProps()}
          >
            <input {...getInputProps()} />
            <img
              src={imageUrl}
              alt="Uploaded preview"
              className="w-full max-h-64 object-contain bg-slate-950/5 dark:bg-slate-950"
            />
            {/* Replace overlay */}
            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-all flex items-center justify-center">
              <span className="opacity-0 group-hover:opacity-100 transition-opacity text-xs text-white font-medium flex items-center gap-1.5 bg-black/50 px-3 py-1.5 rounded-full">
                <Upload className="w-3 h-3" />
                Replace image
              </span>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            {...getRootProps()}
            className={cn(
              'px-6 py-12 flex flex-col items-center justify-center gap-3 cursor-pointer transition-colors',
              isDragActive
                ? 'bg-violet-50 dark:bg-violet-500/10'
                : 'hover:bg-slate-50 dark:hover:bg-slate-800/40',
            )}
          >
            <input {...getInputProps()} />
            <div
              className={cn(
                'w-14 h-14 rounded-2xl flex items-center justify-center transition-colors',
                isDragActive
                  ? 'bg-violet-100 dark:bg-violet-500/20'
                  : 'bg-slate-100 dark:bg-slate-800',
              )}
            >
              <Upload
                className={cn(
                  'w-6 h-6 transition-colors',
                  isDragActive
                    ? 'text-violet-600 dark:text-violet-400'
                    : 'text-slate-400 dark:text-slate-500',
                )}
              />
            </div>
            <div className="text-center">
              <p className={cn(
                'text-sm font-semibold transition-colors',
                isDragActive
                  ? 'text-violet-700 dark:text-violet-300'
                  : 'text-slate-700 dark:text-slate-300',
              )}>
                {isDragActive ? 'Drop to upload' : 'Drag & drop an image'}
              </p>
              <p className="text-xs text-slate-400 dark:text-slate-600 mt-1">
                or click to browse · JPG, PNG, WEBP, BMP
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
