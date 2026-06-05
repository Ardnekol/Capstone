import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from model import AVAILABLE_MODELS, DEFAULT_MODEL, florence
from schemas import HealthResponse, InferenceResponse, ModelsResponse, TaskInfoResponse
from tasks import CASCADED_MAP, TASK_MAP, TASKS
from visualizer import image_to_base64, visualize

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    florence.load(os.getenv("DEFAULT_MODEL", DEFAULT_MODEL))
    yield


app = FastAPI(
    title="Florence-2 Playground API",
    version="1.0.0",
    description="Full-featured REST API for Microsoft Florence-2 vision-language tasks.",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    gpu_name = gpu_total = gpu_used = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_total = round(props.total_memory / 1e9, 2)
        gpu_used = round(torch.cuda.memory_allocated(0) / 1e9, 2)
    return HealthResponse(
        status="ok",
        device=florence.device,
        loaded_models=florence.loaded_models(),
        gpu_name=gpu_name,
        gpu_memory_total_gb=gpu_total,
        gpu_memory_used_gb=gpu_used,
    )


# ── Tasks & Models ─────────────────────────────────────────────────────────────
@app.get("/api/tasks", response_model=list[TaskInfoResponse], tags=["Meta"])
def get_tasks() -> list[TaskInfoResponse]:
    return [
        TaskInfoResponse(
            key=t.key,
            name=t.name,
            group=t.group.value,
            description=t.description,
            example=t.example,
            needs_text_input=t.needs_text_input,
            needs_region_input=t.needs_region_input,
            text_input_placeholder=t.text_input_placeholder,
        )
        for t in TASKS
    ]


@app.get("/api/models", response_model=ModelsResponse, tags=["Meta"])
def get_models() -> ModelsResponse:
    return ModelsResponse(models=AVAILABLE_MODELS, default=DEFAULT_MODEL)


# ── Inference ──────────────────────────────────────────────────────────────────
@app.post("/api/infer", response_model=InferenceResponse, tags=["Inference"])
async def infer(
    image: UploadFile = File(..., description="Image file (JPEG/PNG/WEBP)"),
    task: str = Form(..., description="Task key, e.g. <OD>"),
    model_id: str = Form(DEFAULT_MODEL, description="Florence-2 model variant"),
    text_input: Optional[str] = Form(None, description="Optional text prompt or region coords"),
) -> InferenceResponse:
    # ── Validate inputs ────────────────────────────────────────────────────────
    if task not in TASK_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown task: '{task}'. Call /api/tasks for valid keys.")
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model: '{model_id}'. Call /api/models.")

    task_info = TASK_MAP[task]

    # ── Load image ─────────────────────────────────────────────────────────────
    try:
        raw_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {exc}")

    # ── Normalize region input to Florence loc tokens ──────────────────────────
    final_text = text_input
    if task_info.needs_region_input and text_input:
        try:
            parts = [float(v.strip()) for v in text_input.replace(";", ",").split(",")]
            x1, y1, x2, y2 = parts[:4]
            w, h = pil_image.width, pil_image.height
            lx1 = int(x1 / w * 1000)
            ly1 = int(y1 / h * 1000)
            lx2 = int(x2 / w * 1000)
            ly2 = int(y2 / h * 1000)
            final_text = f"<loc_{lx1}><loc_{ly1}><loc_{lx2}><loc_{ly2}>"
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Region input must be 'x1,y1,x2,y2' in pixel coordinates.",
            )

    # ── Run inference ──────────────────────────────────────────────────────────
    try:
        if task in CASCADED_MAP:
            caption_task = CASCADED_MAP[task]
            caption_result, t1 = florence.run(caption_task, pil_image, model_id=model_id)
            caption_text = caption_result.get(caption_task, "")
            grounding_result, t2 = florence.run(
                "<CAPTION_TO_PHRASE_GROUNDING>",
                pil_image,
                text_input=caption_text,
                model_id=model_id,
            )
            raw_output = {
                "caption": caption_text,
                "<CAPTION_TO_PHRASE_GROUNDING>": grounding_result.get("<CAPTION_TO_PHRASE_GROUNDING>", {}),
            }
            processing_time = t1 + t2
        elif task == "<MULTI_INSTANCE_SEGMENTATION>":
            if not text_input or not text_input.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Multi-instance segmentation requires a text prompt (the object class to segment).",
                )

            # Stage 1: grounding to find every instance
            ground_result, t_ground = florence.run(
                "<CAPTION_TO_PHRASE_GROUNDING>",
                pil_image,
                text_input=text_input,
                model_id=model_id,
            )
            g_data = ground_result.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
            inst_bboxes = g_data.get("bboxes", []) if isinstance(g_data, dict) else []
            inst_labels = g_data.get("labels", []) if isinstance(g_data, dict) else []

            # Stage 2: per-bbox segmentation
            all_polygons: list = []
            all_labels: list[str] = []
            all_instance_bboxes: list = []
            seg_time = 0.0
            w, h = pil_image.width, pil_image.height
            for bbox, label in zip(inst_bboxes, inst_labels):
                x1, y1, x2, y2 = bbox
                lx1 = int(x1 / w * 1000)
                ly1 = int(y1 / h * 1000)
                lx2 = int(x2 / w * 1000)
                ly2 = int(y2 / h * 1000)
                loc_prompt = f"<loc_{lx1}><loc_{ly1}><loc_{lx2}><loc_{ly2}>"
                seg_result, t_seg = florence.run(
                    "<REGION_TO_SEGMENTATION>",
                    pil_image,
                    text_input=loc_prompt,
                    model_id=model_id,
                )
                seg_time += t_seg
                seg_data = seg_result.get("<REGION_TO_SEGMENTATION>", {})
                polys = seg_data.get("polygons", []) if isinstance(seg_data, dict) else []
                if polys:
                    # Each call returns polygon group(s); one instance per call here.
                    all_polygons.append(polys[0] if polys else [])
                    all_labels.append(label)
                    all_instance_bboxes.append(bbox)

            raw_output = {
                "<MULTI_INSTANCE_SEGMENTATION>": {
                    "query": text_input,
                    "count": len(all_polygons),
                    "polygons": all_polygons,
                    "labels": all_labels,
                    "instance_bboxes": all_instance_bboxes,
                    "stage1_detected": len(inst_bboxes),
                }
            }
            processing_time = t_ground + seg_time
        else:
            raw_output, processing_time = florence.run(task, pil_image, final_text, model_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    # ── Visualize ──────────────────────────────────────────────────────────────
    annotated_b64: Optional[str] = None
    try:
        ann = visualize(task, raw_output, pil_image)
        if ann is not None:
            annotated_b64 = image_to_base64(ann)
    except Exception as viz_exc:
        import traceback
        print(f"[Visualize] Error for task {task}:\n{traceback.format_exc()}")

    return InferenceResponse(
        task=task,
        task_name=task_info.name,
        raw_output=raw_output,
        annotated_image=annotated_b64,
        processing_time_ms=round(processing_time, 1),
    )


# ── Serve React build in production ───────────────────────────────────────────
if os.getenv("SERVE_STATIC", "false").lower() == "true":
    static_dir = Path(os.getenv("STATIC_DIR", "../frontend/dist"))
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    _host = os.getenv("HOST", "0.0.0.0")
    _port = int(os.getenv("PORT", 7860))

    config = uvicorn.Config(app, host=_host, port=_port, reload=False)
    server = uvicorn.Server(config)

    # Bind the socket IMMEDIATELY — before the model starts loading.
    # This claims the port in milliseconds and makes any "address in use"
    # error fail fast, rather than after a multi-minute model load.
    sock = config.bind_socket()
    asyncio.run(server.serve(sockets=[sock]))
