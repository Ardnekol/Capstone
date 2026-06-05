import { AnimatePresence, motion } from 'framer-motion';
import { AlertCircle, Clock, Sparkles } from 'lucide-react';
import { BboxOverlay } from './BboxOverlay';
import { ExportButton } from './ExportButton';
import { ImageCompare } from './ImageCompare';
import { JsonViewer } from './JsonViewer';
import { StructuredOutput } from './StructuredOutput';
import { formatMs } from '@/lib/utils';
import type { InferenceResponse } from '@/types';

// Tasks that produce segmentation masks — enable the opacity slider
const SEGMENTATION_TASKS = new Set([
  '<REFERRING_EXPRESSION_SEGMENTATION>',
  '<REGION_TO_SEGMENTATION>',
  '<MULTI_INSTANCE_SEGMENTATION>',
]);

// Extract bbox data from raw_output for detection/grounding tasks
function extractBboxes(
  task: string,
  rawOutput: unknown,
): { bboxes: number[][]; labels: string[] } | null {
  const out = rawOutput as Record<string, unknown>;

  // Cascaded grounding tasks store results under the grounding key
  if (task === '<CAPTION_GROUNDING>' || task === '<DETAILED_CAPTION_GROUNDING>' || task === '<MORE_DETAILED_CAPTION_GROUNDING>') {
    const d = out['<CAPTION_TO_PHRASE_GROUNDING>'] as { bboxes?: number[][]; labels?: string[] } | undefined;
    if (d?.bboxes?.length) return { bboxes: d.bboxes, labels: d.labels ?? [] };
    return null;
  }

  const d = out[task] as { bboxes?: number[][]; labels?: string[]; bboxes_labels?: string[] } | undefined;
  if (d?.bboxes?.length) {
    return { bboxes: d.bboxes, labels: d.labels ?? d.bboxes_labels ?? [] };
  }
  return null;
}

// ── Skeleton ──────────────────────────────────────────────────────────────────
function LoadingSkeleton() {
  return (
    <div className="flex flex-col gap-4">
      <div className="h-72 rounded-xl shimmer" />
      <div className="h-24 rounded-xl shimmer" />
      <div className="space-y-2">
        <div className="h-3 w-2/3 rounded-full shimmer" />
        <div className="h-3 w-1/2 rounded-full shimmer" />
        <div className="h-3 w-3/4 rounded-full shimmer" />
      </div>
    </div>
  );
}

// ── Empty state ───────────────────────────────────────────────────────────────
function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center gap-5 py-20 text-center">
      <div className="relative">
        <div className="w-[72px] h-[72px] rounded-3xl bg-slate-100 dark:bg-slate-800 flex items-center justify-center">
          <Sparkles className="w-8 h-8 text-slate-300 dark:text-slate-600" />
        </div>
        {/* Decorative dots */}
        <span className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-violet-400/30 dark:bg-violet-500/20" />
        <span className="absolute -bottom-1 -left-1 w-2 h-2 rounded-full bg-blue-400/30 dark:bg-blue-500/20" />
      </div>
      <div>
        <p className="font-semibold text-slate-700 dark:text-slate-300">Ready to explore</p>
        <p className="text-sm text-slate-400 dark:text-slate-500 mt-1.5 max-w-[260px] leading-relaxed">
          Upload an image, choose a task, and hit{' '}
          <span className="text-violet-500 font-medium">Run Inference</span> to see
          Florence-2 in action.
        </p>
      </div>
      <div className="flex items-center gap-1.5 text-xs text-slate-300 dark:text-slate-700">
        <kbd className="bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 px-1.5 py-0.5 rounded font-mono text-slate-500 dark:text-slate-400">
          ⌘
        </kbd>
        <kbd className="bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 px-1.5 py-0.5 rounded font-mono text-slate-500 dark:text-slate-400">
          ↵
        </kbd>
        <span>to run</span>
      </div>
    </div>
  );
}

// ── Props ─────────────────────────────────────────────────────────────────────
interface Props {
  result: InferenceResponse | null;
  loading: boolean;
  error: string | null;
  originalImage: string | null; // object URL of the uploaded file
}

export function ResultPanel({ result, loading, error, originalImage }: Props) {
  const isSegTask = result ? SEGMENTATION_TASKS.has(result.task) : false;
  const bboxData = result ? extractBboxes(result.task, result.raw_output) : null;

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 flex flex-col h-full min-h-[520px]">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-slate-100 dark:border-slate-800 shrink-0">
        <span className="text-[10px] font-semibold tracking-widest text-slate-400 dark:text-slate-600 uppercase">
          Output
        </span>

        {result && (
          <div className="flex items-center gap-2">
            <span className="flex items-center gap-1 text-xs text-slate-400 dark:text-slate-500">
              <Clock className="w-3 h-3" />
              {formatMs(result.processing_time_ms)}
            </span>
            <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-violet-50 dark:bg-violet-500/10 text-violet-600 dark:text-violet-400 border border-violet-200 dark:border-violet-500/20">
              {result.task_name}
            </span>
            <ExportButton result={result} />
          </div>
        )}
      </div>

      {/* Body */}
      <div className="flex-1 p-4 overflow-y-auto scrollbar-thin">
        <AnimatePresence mode="wait">
          {/* Loading */}
          {loading && (
            <motion.div
              key="loading"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <LoadingSkeleton />
            </motion.div>
          )}

          {/* Error */}
          {!loading && error && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex items-start gap-3 p-4 rounded-xl bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/30"
            >
              <AlertCircle className="w-4 h-4 mt-0.5 shrink-0 text-rose-500" />
              <div>
                <p className="font-semibold text-sm text-rose-700 dark:text-rose-300">
                  Inference Error
                </p>
                <p className="text-sm mt-0.5 text-rose-600 dark:text-rose-400 opacity-90">
                  {error}
                </p>
              </div>
            </motion.div>
          )}

          {/* Empty */}
          {!loading && !error && !result && (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <EmptyState />
            </motion.div>
          )}

          {/* Result */}
          {!loading && !error && result && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25 }}
              className="flex flex-col gap-4"
            >
              {/* ── Visual result ── */}
              {bboxData && originalImage ? (
                // Detection/grounding: animated bbox draw-in overlay
                <BboxOverlay
                  imageUrl={originalImage}
                  bboxes={bboxData.bboxes}
                  labels={bboxData.labels}
                />
              ) : result.annotated_image && originalImage ? (
                // Segmentation / OCR: side-by-side compare slider
                <ImageCompare
                  original={originalImage}
                  annotated={result.annotated_image}
                  showOpacityMode={isSegTask}
                />
              ) : result.annotated_image ? (
                <img
                  src={`data:image/png;base64,${result.annotated_image}`}
                  alt="Annotated result"
                  className="w-full object-contain max-h-[420px] rounded-xl border border-slate-200 dark:border-slate-700"
                />
              ) : originalImage ? (
                // No annotation produced — show original with a notice
                <div className="relative rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700">
                  <img
                    src={originalImage}
                    alt="Original"
                    className="w-full object-contain max-h-[420px] opacity-60"
                  />
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-slate-900/50 dark:bg-slate-900/70">
                    <p className="text-white text-sm font-semibold">No visual output</p>
                    <p className="text-slate-300 text-xs text-center px-6">
                      {isSegTask
                        ? 'The model could not segment that object. Try a more specific prompt or a clearer image.'
                        : 'No visual annotation was produced for this task.'}
                    </p>
                  </div>
                </div>
              ) : null}

              {/* ── Structured smart output ── */}
              <StructuredOutput task={result.task} rawOutput={result.raw_output} />

              {/* ── Raw JSON (collapsed by default when there's a visual) ── */}
              <JsonViewer
                data={result.raw_output}
                defaultOpen={!result.annotated_image}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
