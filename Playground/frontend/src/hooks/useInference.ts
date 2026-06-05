import { useCallback, useState } from 'react';
import { runInference } from '@/lib/api';
import type { InferenceResponse } from '@/types';

interface Params {
  image: File | null;
  task: string;
  modelId: string;
  textInput: string;
}

export function useInference({ image, task, modelId, textInput }: Params) {
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async (): Promise<InferenceResponse | null> => {
    if (!image || !task) return null;
    setLoading(true);
    setError(null);
    try {
      const data = await runInference({
        image,
        task,
        model_id: modelId,
        text_input: textInput.trim() || undefined,
      });
      setResult(data);
      return data;
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })
        ?.response?.data?.detail;
      setError(detail ?? (err instanceof Error ? err.message : 'Inference failed'));
      return null;
    } finally {
      setLoading(false);
    }
  }, [image, task, modelId, textInput]);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, loading, error, run, reset };
}
