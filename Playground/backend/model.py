import os
import time
from typing import Optional

import torch
from PIL import Image

# ── flash_attn workaround ──────────────────────────────────────────────────────
# Recent Florence-2 hub files unconditionally import flash_attn, but the model
# works fine without it using attn_implementation="eager".
# check_imports() is called inside from_pretrained(), so the patch must be
# active during the load() call, not just at module import time.
import transformers.dynamic_module_utils as _dmu

_orig_check_imports = _dmu.check_imports


def _patched_check_imports(filename: str, *args, **kwargs):
    try:
        return _orig_check_imports(filename, *args, **kwargs)
    except ImportError as exc:
        if "flash_attn" in str(exc):
            return []  # treat flash_attn as optional
        raise


from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: E402

AVAILABLE_MODELS = [
    "microsoft/Florence-2-large",
    "microsoft/Florence-2-large-ft",
]

DEFAULT_MODEL = "microsoft/Florence-2-large"

# Allow operators to pin the model cache to a specific directory (e.g. fast NVMe)
_CACHE_DIR: Optional[str] = os.getenv("MODEL_CACHE_DIR") or None

# Allow overriding dtype via env var for servers where bfloat16 is preferred
_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


class FlorenceModel:
    def __init__(self) -> None:
        self._models: dict = {}
        self._processors: dict = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        _env_dtype = os.getenv("TORCH_DTYPE", "").lower()
        if _env_dtype in _DTYPE_MAP:
            self.dtype = _DTYPE_MAP[_env_dtype]
        else:
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    def load(self, model_id: str = DEFAULT_MODEL) -> None:
        if model_id in self._models:
            return
        print(f"[Florence] Loading {model_id} on {self.device} ({self.dtype})...")
        _dmu.check_imports = _patched_check_imports
        try:
            self._models[model_id] = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                cache_dir=_CACHE_DIR,
                attn_implementation="eager",  # avoid flash_attn at runtime
            ).to(self.device).eval()
            self._processors[model_id] = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=_CACHE_DIR,
            )
        finally:
            _dmu.check_imports = _orig_check_imports
        print(f"[Florence] {model_id} ready.")

    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    def run(
        self,
        task_prompt: str,
        image: Image.Image,
        text_input: Optional[str] = None,
        model_id: str = DEFAULT_MODEL,
    ) -> tuple[dict, float]:
        if model_id not in self._models:
            self.load(model_id)

        model = self._models[model_id]
        processor = self._processors[model_id]

        prompt = task_prompt if text_input is None else task_prompt + text_input

        inputs = processor(text=prompt, images=image, return_tensors="pt")
        # Cast pixel_values to model dtype; keep input_ids as long
        inputs = {
            k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device))
            for k, v in inputs.items()
        }

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=8192,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )
        return parsed, elapsed_ms


# Module-level singleton — loaded once, reused across requests
florence = FlorenceModel()
