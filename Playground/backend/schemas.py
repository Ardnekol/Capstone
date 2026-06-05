from pydantic import BaseModel
from typing import Any, Optional


class InferenceResponse(BaseModel):
    task: str
    task_name: str
    raw_output: Any
    annotated_image: Optional[str] = None  # base64-encoded PNG, None for text-only tasks
    processing_time_ms: float


class TaskInfoResponse(BaseModel):
    key: str
    name: str
    group: str
    description: str
    example: str
    needs_text_input: bool
    needs_region_input: bool
    text_input_placeholder: Optional[str]


class ModelsResponse(BaseModel):
    models: list[str]
    default: str


class HealthResponse(BaseModel):
    status: str
    device: str
    loaded_models: list[str]
    gpu_name: Optional[str]
    gpu_memory_total_gb: Optional[float]
    gpu_memory_used_gb: Optional[float]
