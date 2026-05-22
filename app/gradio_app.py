import time
import numpy as np
import torch
import gradio as gr
from PIL import Image as PILImage

from app.registry import registry
from app.models.clip_model import rank_masks_by_prompt
from app.utils import apply_mask_overlay, draw_bbox

COLORS = [(0, 120, 255), (255, 80, 0), (0, 200, 80)]
MAX_DIM = 1024  # SAM is trained at 1024px; larger inputs just waste VRAM

MODEL_CHOICES = ["sam1", "sam2", "sam3"]
AVAILABLE = [m for m in MODEL_CHOICES if registry.is_available(m)]


def _cap_image(img_np: np.ndarray) -> np.ndarray:
    h, w = img_np.shape[:2]
    if max(h, w) <= MAX_DIM:
        return img_np
    scale = MAX_DIM / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    pil = PILImage.fromarray(img_np).resize((new_w, new_h), PILImage.LANCZOS)
    return np.array(pil)


def _run_inference(img_np: np.ndarray, prompt: str, model_version: str, top_k: int):
    """Run one model and return (annotated_image, metrics_dict, masks_list)."""
    model = registry.get(model_version)
    # Restore to GPU in case a previous compare run offloaded it to CPU
    target = "cuda" if torch.cuda.is_available() else "cpu"
    if model.device != target:
        model.move_to(target)
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    t0 = time.perf_counter()

    if model_version == "sam3":
        masks = model.segment_with_text(img_np, prompt)
        masks = masks[:top_k]
    else:
        raw_masks = model.generate_masks(img_np)
        masks = rank_masks_by_prompt(img_np, raw_masks, prompt, top_k=top_k)

    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    peak_mb = 0.0
    if torch.cuda.is_available():
        peak_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
        torch.cuda.empty_cache()

    # annotate
    result = img_np.copy()
    for i, m in enumerate(masks):
        color = COLORS[i % len(COLORS)]
        result = apply_mask_overlay(result, m, color=color)
        result = draw_bbox(result, m, color=color)

    metrics = {
        "model": model_version,
        "device": registry.status()[model_version],
        "inference_ms": inference_ms,
        "peak_gpu_mb": peak_mb,
        "mask_count": len(masks),
    }

    return PILImage.fromarray(result), metrics, masks


def segment_single(image, prompt, model_version, top_k):
    if image is None:
        return None, "Upload an image first.", ""
    if not prompt.strip():
        return None, "Enter a text prompt.", ""
    if model_version not in AVAILABLE:
        return None, f"{model_version} is not available.", ""

    img_np = _cap_image(np.array(image.convert("RGB")))
    annotated, metrics, masks = _run_inference(img_np, prompt, model_version, int(top_k))

    metrics_md = (
        f"**Model:** {metrics['model']} ({metrics['device']})  \n"
        f"**Inference:** {metrics['inference_ms']} ms  |  "
        f"**Peak VRAM:** {metrics['peak_gpu_mb']} MB  |  "
        f"**Masks found:** {metrics['mask_count']}"
    )

    rows = "\n".join(
        f"| {i+1} | {m['score']:.4f} | {m['area']:,} | {[round(v) for v in m['bbox']]} |"
        for i, m in enumerate(masks)
    )
    table_md = (
        "| Rank | Score | Area (px) | BBox (xywh) |\n"
        "|------|-------|-----------|-------------|\n"
        + rows
    )

    return annotated, metrics_md, table_md


def segment_compare(image, prompt, top_k):
    if image is None:
        return None, None, None, "Upload an image first."
    if not prompt.strip():
        return None, None, None, "Enter a text prompt."

    img_np = _cap_image(np.array(image.convert("RGB")))
    results = {}

    for model_version in AVAILABLE:
        annotated, metrics, _ = _run_inference(img_np, prompt, model_version, int(top_k))
        results[model_version] = (annotated, metrics)
        # Offload to CPU before loading the next model to avoid OOM on 8GB VRAM
        registry.get(model_version).move_to("cpu")
        torch.cuda.empty_cache()

    def get(name):
        return results[name][0] if name in results else None

    summary_rows = []
    for name, (_, m) in results.items():
        summary_rows.append(
            f"| {name} | {m['device']} | {m['inference_ms']} ms | "
            f"{m['peak_gpu_mb']} MB | {m['mask_count']} |"
        )
    summary_md = (
        "| Model | Device | Inference | Peak VRAM | Masks |\n"
        "|-------|--------|-----------|-----------|-------|\n"
        + "\n".join(summary_rows)
    )

    return get("sam1"), get("sam2"), get("sam3"), summary_md


# ── UI ──────────────────────────────────────────────────────────────────────

with gr.Blocks(title="SAM Benchmark Sandbox") as demo:
    gr.Markdown("# SAM Benchmark Sandbox\nCompare SAM1 · SAM2 · SAM3 zero-shot text segmentation")

    with gr.Tabs():

        # ── Tab 1: Single model ──────────────────────────────────────────────
        with gr.Tab("Single Model"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_in = gr.Image(type="pil", label="Input Image")
                    prompt_in = gr.Textbox(label="Text Prompt", placeholder="e.g. dog, car, person")
                    model_dd = gr.Dropdown(
                        choices=AVAILABLE,
                        value=AVAILABLE[0] if AVAILABLE else None,
                        label="Model",
                    )
                    topk_sl = gr.Slider(1, 5, value=3, step=1, label="Top-K masks")
                    run_btn = gr.Button("Segment", variant="primary")

                with gr.Column(scale=1):
                    img_out = gr.Image(type="pil", label="Segmented Output")
                    metrics_out = gr.Markdown(label="Metrics")
                    table_out = gr.Markdown(label="Mask Details")

            run_btn.click(
                fn=segment_single,
                inputs=[img_in, prompt_in, model_dd, topk_sl],
                outputs=[img_out, metrics_out, table_out],
            )

        # ── Tab 2: Compare all models ────────────────────────────────────────
        with gr.Tab("Compare All Models"):
            with gr.Row():
                cmp_img_in = gr.Image(type="pil", label="Input Image")
                with gr.Column():
                    cmp_prompt = gr.Textbox(label="Text Prompt", placeholder="e.g. dog, car, person")
                    cmp_topk = gr.Slider(1, 5, value=3, step=1, label="Top-K masks")
                    cmp_btn = gr.Button("Compare", variant="primary")

            with gr.Row():
                out_sam1 = gr.Image(type="pil", label="SAM1 + CLIP")
                out_sam2 = gr.Image(type="pil", label="SAM2 + CLIP")
                out_sam3 = gr.Image(type="pil", label="SAM3 (native text)")

            cmp_summary = gr.Markdown(label="Benchmark Summary")

            cmp_btn.click(
                fn=segment_compare,
                inputs=[cmp_img_in, cmp_prompt, cmp_topk],
                outputs=[out_sam1, out_sam2, out_sam3, cmp_summary],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
