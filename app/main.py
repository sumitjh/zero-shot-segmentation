import io
import time
import base64

import numpy as np
import torch
from PIL import Image as PILImage
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.registry import registry
from app.models.clip_model import rank_masks_by_prompt
from app.utils import apply_mask_overlay, draw_bbox, masks_to_response

app = FastAPI(title="SAM Benchmark Sandbox")


def _image_to_base64(image: np.ndarray) -> str:
    pil = PILImage.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/models")
def list_models():
    return registry.status()


@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    prompt: str = Form(""),
    model_version: str = Form("sam1"),
    prompt_type: str = Form("text"),  # text | auto
    top_k: int = Form(3),
    mask_return: bool = Form(False),
):
    # validate
    if model_version not in ("sam1", "sam2", "sam3"):
        raise HTTPException(400, "model_version must be sam1, sam2, or sam3")
    if prompt_type not in ("text", "auto"):
        raise HTTPException(400, "prompt_type must be text or auto")
    if prompt_type == "auto" and model_version == "sam3":
        raise HTTPException(400, "SAM3 does not support auto prompt_type — use text")
    if prompt_type == "text" and not prompt:
        raise HTTPException(400, "prompt is required for text prompt_type")
    if not registry.is_available(model_version):
        raise HTTPException(503, f"{model_version} is {registry.status()[model_version]}")

    # load image
    raw = await image.read()
    img_np = np.array(PILImage.open(io.BytesIO(raw)).convert("RGB"))

    model = registry.get(model_version)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    t0 = time.perf_counter()

    if model_version == "sam3":
        masks = model.segment_with_text(img_np, prompt)
        masks = masks[:top_k]
    else:
        raw_masks = model.generate_masks(img_np)
        if prompt_type == "text":
            masks = rank_masks_by_prompt(img_np, raw_masks, prompt, top_k=top_k)
        else:
            masks = raw_masks[:top_k]
            for m in masks:
                m["score"] = 0.0

    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    peak_mb = 0.0
    if torch.cuda.is_available():
        peak_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
        torch.cuda.empty_cache()

    # build annotated image
    colors = [(0, 120, 255), (255, 80, 0), (0, 200, 80)]
    result = img_np.copy()
    for i, m in enumerate(masks):
        color = colors[i % len(colors)]
        result = apply_mask_overlay(result, m, color=color)
        result = draw_bbox(result, m, color=color)

    response = {
        "model": model_version,
        "device": registry.status()[model_version],
        "prompt_type": prompt_type,
        "metrics": {
            "inference_ms": inference_ms,
            "mask_count": len(masks),
            "peak_gpu_memory_mb": peak_mb,
        },
        "masks": masks_to_response(masks, include_rle=mask_return),
        "image_b64": _image_to_base64(result),
    }
    return JSONResponse(response)
