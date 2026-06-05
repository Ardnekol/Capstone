import { useState } from 'react';
import { Check, Copy, Hash, Tag } from 'lucide-react';

// ── Helpers ───────────────────────────────────────────────────────────────────

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };
  return (
    <button
      onClick={handleCopy}
      className="flex items-center gap-1 px-2 py-1 rounded-md text-xs text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
    >
      {copied ? <Check className="w-3 h-3 text-emerald-500" /> : <Copy className="w-3 h-3" />}
      {copied ? 'Copied' : 'Copy'}
    </button>
  );
}

function LabelCloud({ labels }: { labels: string[] }) {
  const counts: Record<string, number> = {};
  labels.forEach((l) => { counts[l] = (counts[l] ?? 0) + 1; });
  const unique = Object.entries(counts).sort((a, b) => b[1] - a[1]);

  return (
    <div className="flex flex-wrap gap-1.5">
      {unique.map(([label, count]) => (
        <span
          key={label}
          className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700"
        >
          {label}
          {count > 1 && (
            <span className="text-[10px] font-bold text-violet-500">×{count}</span>
          )}
        </span>
      ))}
    </div>
  );
}

// ── Per-task renderers ────────────────────────────────────────────────────────

function CaptionOutput({ text }: { text: string }) {
  return (
    <div className="rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 p-4">
      <div className="flex items-start justify-between gap-2 mb-2">
        <span className="text-[10px] font-semibold tracking-widest text-slate-400 uppercase">
          Caption
        </span>
        <CopyButton text={text} />
      </div>
      <p className="text-slate-800 dark:text-slate-100 text-base leading-relaxed font-medium">
        "{text}"
      </p>
      <p className="text-xs text-slate-400 dark:text-slate-600 mt-2">
        {text.split(/\s+/).filter(Boolean).length} words · {text.length} chars
      </p>
    </div>
  );
}

function DetectionOutput({ labels }: { labels: string[] }) {
  return (
    <div className="rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 p-4">
      <div className="flex items-center gap-2 mb-3">
        <Tag className="w-3.5 h-3.5 text-blue-500" />
        <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">
          {labels.length} object{labels.length !== 1 ? 's' : ''} detected
        </span>
      </div>
      <LabelCloud labels={labels} />
    </div>
  );
}

function OcrOutput({ text }: { text: string }) {
  return (
    <div className="rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 p-4">
      <div className="flex items-start justify-between gap-2 mb-2">
        <span className="text-[10px] font-semibold tracking-widest text-slate-400 uppercase">
          Extracted Text
        </span>
        <CopyButton text={text} />
      </div>
      <pre className="font-mono text-sm text-slate-800 dark:text-slate-100 whitespace-pre-wrap break-words leading-relaxed">
        {text || <span className="italic text-slate-400">No text found</span>}
      </pre>
    </div>
  );
}

function OcrRegionOutput({ labels }: { labels: string[] }) {
  return (
    <div className="rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 p-4">
      <div className="flex items-center gap-2 mb-3">
        <Hash className="w-3.5 h-3.5 text-yellow-500" />
        <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">
          {labels.length} text region{labels.length !== 1 ? 's' : ''} found
        </span>
        <CopyButton text={labels.join('\n')} />
      </div>
      <div className="flex flex-col gap-1 max-h-40 overflow-y-auto scrollbar-thin">
        {labels.map((l, i) => (
          <span key={i} className="text-xs font-mono text-slate-600 dark:text-slate-300 bg-slate-100 dark:bg-slate-800 px-2 py-1 rounded">
            {l}
          </span>
        ))}
      </div>
    </div>
  );
}

function GroundingOutput({ labels, caption }: { labels: string[]; caption?: string }) {
  return (
    <div className="rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 p-4">
      {caption && (
        <p className="text-sm text-slate-700 dark:text-slate-200 italic mb-3 pb-3 border-b border-slate-200 dark:border-slate-700">
          "{caption}"
        </p>
      )}
      <div className="flex items-center gap-2 mb-2">
        <span className="text-[10px] font-semibold tracking-widest text-slate-400 uppercase">
          Grounded Phrases
        </span>
        <span className="text-xs text-slate-400">{labels.length} regions</span>
      </div>
      <LabelCloud labels={labels} />
    </div>
  );
}

function SegmentationOutput({ labels }: { labels: string[] }) {
  return (
    <div className="rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 p-4">
      <div className="flex items-center gap-2 mb-3">
        <span className="w-3 h-3 rounded-full bg-orange-400 shrink-0" />
        <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">
          {labels.length} segment{labels.length !== 1 ? 's' : ''} found
        </span>
      </div>
      {labels.length > 0 && <LabelCloud labels={labels} />}
    </div>
  );
}

function SingleTextOutput({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 p-4">
      <div className="flex items-start justify-between gap-2 mb-1">
        <span className="text-[10px] font-semibold tracking-widest text-slate-400 uppercase">
          {label}
        </span>
        <CopyButton text={value} />
      </div>
      <p className="text-slate-800 dark:text-slate-100 text-sm leading-relaxed">{value}</p>
    </div>
  );
}

// ── Main router ───────────────────────────────────────────────────────────────

interface Props {
  task: string;
  rawOutput: unknown;
}

export function StructuredOutput({ task, rawOutput }: Props) {
  const out = rawOutput as Record<string, unknown>;

  // ── Captions ───────────────────────────────────────────────────────────────
  if (task === '<CAPTION>' && typeof out['<CAPTION>'] === 'string')
    return <CaptionOutput text={out['<CAPTION>']} />;

  if (task === '<DETAILED_CAPTION>' && typeof out['<DETAILED_CAPTION>'] === 'string')
    return <CaptionOutput text={out['<DETAILED_CAPTION>']} />;

  if (task === '<MORE_DETAILED_CAPTION>' && typeof out['<MORE_DETAILED_CAPTION>'] === 'string')
    return <CaptionOutput text={out['<MORE_DETAILED_CAPTION>']} />;

  // ── Detection ──────────────────────────────────────────────────────────────
  if (
    task === '<OD>' &&
    typeof out['<OD>'] === 'object' &&
    (out['<OD>'] as Record<string, unknown>)?.labels
  ) {
    const labels = (out['<OD>'] as { labels: string[] }).labels;
    return <DetectionOutput labels={labels} />;
  }

  if (
    task === '<DENSE_REGION_CAPTION>' &&
    typeof out['<DENSE_REGION_CAPTION>'] === 'object'
  ) {
    const labels = (out['<DENSE_REGION_CAPTION>'] as { labels: string[] }).labels ?? [];
    return <DetectionOutput labels={labels} />;
  }

  if (task === '<REGION_PROPOSAL>') {
    const data = out['<REGION_PROPOSAL>'] as { bboxes?: unknown[] } | undefined;
    const count = data?.bboxes?.length ?? 0;
    return (
      <div className="rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 p-4">
        <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">
          {count} region proposal{count !== 1 ? 's' : ''} found
        </p>
      </div>
    );
  }

  if (
    task === '<OPEN_VOCABULARY_DETECTION>' &&
    typeof out['<OPEN_VOCABULARY_DETECTION>'] === 'object'
  ) {
    const d = out['<OPEN_VOCABULARY_DETECTION>'] as {
      bboxes_labels?: string[];
      labels?: string[];
    };
    const labels = d.bboxes_labels ?? d.labels ?? [];
    return <DetectionOutput labels={labels} />;
  }

  // ── Grounding ──────────────────────────────────────────────────────────────
  if (
    task === '<CAPTION_TO_PHRASE_GROUNDING>' &&
    typeof out['<CAPTION_TO_PHRASE_GROUNDING>'] === 'object'
  ) {
    const labels = (out['<CAPTION_TO_PHRASE_GROUNDING>'] as { labels: string[] }).labels ?? [];
    return <GroundingOutput labels={labels} />;
  }

  if (task === '<REGION_TO_CATEGORY>' && typeof out['<REGION_TO_CATEGORY>'] === 'string')
    return <SingleTextOutput label="Category" value={out['<REGION_TO_CATEGORY>']} />;

  if (task === '<REGION_TO_DESCRIPTION>' && typeof out['<REGION_TO_DESCRIPTION>'] === 'string')
    return <SingleTextOutput label="Description" value={out['<REGION_TO_DESCRIPTION>']} />;

  // ── Segmentation ───────────────────────────────────────────────────────────
  if (
    task === '<REFERRING_EXPRESSION_SEGMENTATION>' &&
    typeof out['<REFERRING_EXPRESSION_SEGMENTATION>'] === 'object'
  ) {
    const labels = (out['<REFERRING_EXPRESSION_SEGMENTATION>'] as { labels: string[] }).labels ?? [];
    return <SegmentationOutput labels={labels} />;
  }

  if (
    task === '<REGION_TO_SEGMENTATION>' &&
    typeof out['<REGION_TO_SEGMENTATION>'] === 'object'
  ) {
    const labels = (out['<REGION_TO_SEGMENTATION>'] as { labels: string[] }).labels ?? [];
    return <SegmentationOutput labels={labels} />;
  }

  if (
    task === '<MULTI_INSTANCE_SEGMENTATION>' &&
    typeof out['<MULTI_INSTANCE_SEGMENTATION>'] === 'object'
  ) {
    const d = out['<MULTI_INSTANCE_SEGMENTATION>'] as {
      query?: string;
      count?: number;
      labels?: string[];
      stage1_detected?: number;
    };
    const labels = d.labels ?? [];
    const detected = d.stage1_detected ?? labels.length;
    const segmented = d.count ?? labels.length;
    return (
      <div className="rounded-xl bg-slate-50 dark:bg-slate-800/40 border border-slate-200 dark:border-slate-700 p-4">
        <div className="flex items-center gap-2 mb-3">
          <span className="w-3 h-3 rounded-full bg-orange-400 shrink-0" />
          <span className="text-sm font-semibold text-slate-700 dark:text-slate-200">
            {segmented} instance{segmented !== 1 ? 's' : ''} segmented
            {detected !== segmented && (
              <span className="ml-1 text-xs text-slate-400 font-normal">
                (from {detected} detected)
              </span>
            )}
          </span>
        </div>
        {d.query && (
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-3">
            Query: <span className="font-mono text-slate-700 dark:text-slate-200">"{d.query}"</span>
          </p>
        )}
        {labels.length > 0 && <LabelCloud labels={labels} />}
      </div>
    );
  }

  // ── OCR ────────────────────────────────────────────────────────────────────
  if (task === '<OCR>' && typeof out['<OCR>'] === 'string')
    return <OcrOutput text={out['<OCR>']} />;

  if (
    task === '<OCR_WITH_REGION>' &&
    typeof out['<OCR_WITH_REGION>'] === 'object'
  ) {
    const labels = (out['<OCR_WITH_REGION>'] as { labels: string[] }).labels ?? [];
    return <OcrRegionOutput labels={labels} />;
  }

  // ── Cascaded ───────────────────────────────────────────────────────────────
  if (
    task.endsWith('_GROUNDING>') &&
    typeof out['<CAPTION_TO_PHRASE_GROUNDING>'] === 'object'
  ) {
    const caption = typeof out.caption === 'string' ? out.caption : undefined;
    const labels =
      (out['<CAPTION_TO_PHRASE_GROUNDING>'] as { labels?: string[] }).labels ?? [];
    return <GroundingOutput labels={labels} caption={caption} />;
  }

  // ── Fallback ───────────────────────────────────────────────────────────────
  return null;
}
