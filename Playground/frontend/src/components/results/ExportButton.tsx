import { Download } from 'lucide-react';
import { useToast } from '@/components/ui/Toast';
import { downloadBlob } from '@/lib/utils';
import type { InferenceResponse } from '@/types';

interface Props {
  result: InferenceResponse;
}

function base64ToBlob(b64: string, mimeType: string): Blob {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mimeType });
}

export function ExportButton({ result }: Props) {
  const { toast } = useToast();

  function handleExport() {
    const ts = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-');
    const taskSlug = result.task.replace(/[<>]/g, '');

    downloadBlob(
      new Blob([JSON.stringify(result.raw_output, null, 2)], { type: 'application/json' }),
      `${taskSlug}_${ts}.json`,
    );

    const files = [`${taskSlug}_${ts}.json`];

    if (result.annotated_image) {
      downloadBlob(
        base64ToBlob(result.annotated_image, 'image/png'),
        `${taskSlug}_${ts}.png`,
      );
      files.push(`${taskSlug}_${ts}.png`);
    }

    toast({
      type: 'success',
      title: 'Export downloaded',
      message: files.join('  +  '),
    });
  }

  return (
    <button
      onClick={handleExport}
      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 transition-colors"
    >
      <Download className="w-3.5 h-3.5" />
      Export
    </button>
  );
}
