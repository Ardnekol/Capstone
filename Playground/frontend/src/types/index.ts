export type TaskGroupId =
  | 'caption'
  | 'detection'
  | 'grounding'
  | 'segmentation'
  | 'ocr'
  | 'cascaded';

export interface TaskInfo {
  key: string;
  name: string;
  group: TaskGroupId;
  description: string;
  example: string;
  needs_text_input: boolean;
  needs_region_input: boolean;
  text_input_placeholder: string | null;
}

export interface InferenceResponse {
  task: string;
  task_name: string;
  raw_output: unknown;
  annotated_image: string | null; // base64 PNG
  processing_time_ms: number;
}

export interface HistoryItem {
  id: string;
  timestamp: number;
  task_key: string;
  task_name: string;
  group: TaskGroupId;
  thumbnail: string; // base64 JPEG
  result: InferenceResponse;
}

export interface HealthStatus {
  status: string;
  device: string;
  loaded_models: string[];
  gpu_name: string | null;
  gpu_memory_total_gb: number | null;
  gpu_memory_used_gb: number | null;
}
