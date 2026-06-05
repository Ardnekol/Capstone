import axios from 'axios';
import type { HealthStatus, InferenceResponse, TaskInfo } from '@/types';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? '/api',
  timeout: 180_000, // 3 min — large models can be slow
});

export async function fetchTasks(): Promise<TaskInfo[]> {
  const { data } = await api.get<TaskInfo[]>('/tasks');
  return data;
}

export async function fetchModels(): Promise<{ models: string[]; default: string }> {
  const { data } = await api.get<{ models: string[]; default: string }>('/models');
  return data;
}

export async function fetchHealth(): Promise<HealthStatus> {
  const { data } = await api.get<HealthStatus>('/health');
  return data;
}

export async function runInference(params: {
  image: File;
  task: string;
  model_id: string;
  text_input?: string;
}): Promise<InferenceResponse> {
  const form = new FormData();
  form.append('image', params.image);
  form.append('task', params.task);
  form.append('model_id', params.model_id);
  if (params.text_input) form.append('text_input', params.text_input);
  const { data } = await api.post<InferenceResponse>('/infer', form);
  return data;
}
