import time
import numpy as np
import torch
import gradio as gr
from PIL import Image as PILImage
from huggingface_hub import hf_hub_download

try:
    import spaces
    ZEROGPU = True
except ImportError:
    ZEROGPU = False

from app.models.sam1_model import SAM1Model
from app.models.sam2_model import SAM2Model
from app.models.sam3_model import SAM3Model
from app.models.clip_model import rank_masks_by_prompt
from app.utils import apply_mask_overlay, draw_bbox

COLORS = [(0, 120, 255), (255, 80, 0), (0, 200, 80)]
MAX_DIM = 1024

# ── Model loading ─────────────────────────────────────────────────────────────
# Download weights to HF cache and build all models on CPU at startup.
# GPU is only used during inference inside @spaces.GPU.

print("Downloading model weights…")
_sam1_ckpt = hf_hub_download("sumitjh/sam-benchmark-checkpoints", "sam_vit_b_01ec64.pth")
_sam2_ckpt = hf_hub_download("facebook/sam2-hiera-base-plus", "sam2_hiera_base_plus.pt")
_sam3_ckpt = hf_hub_download("facebook/sam3.1", "sam3.1_multiplex.pt")
print("Building models on CPU…")
_models = {
    "sam1": SAM1Model(_sam1_ckpt, device="cpu"),
    "sam2": SAM2Model(_sam2_ckpt, device="cpu"),
    "sam3": SAM3Model(checkpoint_path=_sam3_ckpt, bpe_path=None, device="cpu"),
}
print("Models ready.")

MODEL_NAMES = list(_models.keys())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cap_image(img_np: np.ndarray) -> np.ndarray:
    h, w = img_np.shape[:2]
    if max(h, w) <= MAX_DIM:
        return img_np
    scale = MAX_DIM / max(h, w)
    pil = PILImage.fromarray(img_np).resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
    return np.array(pil)


def _run_inference(img_np: np.ndarray, prompt: str, model_name: str, top_k: int):
    model = _models[model_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model.device != device:
        model.move_to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    if model_name == "sam3":
        masks = model.segment_with_text(img_np, prompt)
        masks = masks[:top_k]
        for m in masks:
            m["clip_score"] = m.pop("score")
    else:
        raw_masks = model.generate_masks(img_np)
        masks = rank_masks_by_prompt(img_np, raw_masks, prompt, top_k=top_k)

    inference_ms = round((time.perf_counter() - t0) * 1000, 1)
    peak_mb = 0.0
    if device == "cuda":
        peak_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
        torch.cuda.empty_cache()
        model.move_to("cpu")

    result = img_np.copy()
    for i, m in enumerate(masks):
        color = COLORS[i % len(COLORS)]
        result = apply_mask_overlay(result, m, color=color)
        result = draw_bbox(result, m, color=color)

    metrics = {
        "model": model_name,
        "inference_ms": inference_ms,
        "peak_gpu_mb": peak_mb,
        "mask_count": len(masks),
    }
    return PILImage.fromarray(result), metrics, masks


# ── Inference functions (wrapped with @spaces.GPU when available) ─────────────

def _segment_single_impl(image, prompt, model_name, top_k):
    if image is None:
        return None, "Upload an image first.", ""
    if not prompt.strip():
        return None, "Enter a text prompt.", ""

    img_np = _cap_image(np.array(image.convert("RGB")))
    annotated, metrics, masks = _run_inference(img_np, prompt, model_name, int(top_k))

    metrics_md = (
        f"**Model:** {metrics['model']}  \n"
        f"**Inference:** {metrics['inference_ms']} ms  |  "
        f"**Peak VRAM:** {metrics['peak_gpu_mb']} MB  |  "
        f"**Masks found:** {metrics['mask_count']}"
    )
    rows = "\n".join(
        f"| {i+1} | {m['clip_score']:.4f} | {m['area']:,} | {[round(v) for v in m['bbox']]} |"
        for i, m in enumerate(masks)
    )
    table_md = (
        "| Rank | Score | Area (px) | BBox (xywh) |\n"
        "|------|-------|-----------|-------------|\n"
        + rows
    )
    return annotated, metrics_md, table_md


def _segment_compare_impl(image, prompt, top_k):
    if image is None:
        return None, None, None, "Upload an image first."
    if not prompt.strip():
        return None, None, None, "Enter a text prompt."

    img_np = _cap_image(np.array(image.convert("RGB")))
    results = {}

    for name in MODEL_NAMES:
        try:
            annotated, metrics, _ = _run_inference(img_np, prompt, name, int(top_k))
            results[name] = (annotated, metrics)
        except Exception as e:
            results[name] = (None, {"model": name, "inference_ms": 0,
                                     "peak_gpu_mb": 0, "mask_count": 0, "error": str(e)})

    def get_img(name):
        return results.get(name, (None, {}))[0]

    rows = []
    for name, (_, m) in results.items():
        err = m.get("error", "")
        if err:
            rows.append(f"| {name} | — | — | — | ERROR: {err[:50]} |")
        else:
            rows.append(f"| {name} | {m['inference_ms']} ms | {m['peak_gpu_mb']} MB | {m['mask_count']} |")
    summary_md = (
        "| Model | Inference | Peak VRAM | Masks |\n"
        "|-------|-----------|-----------|-------|\n"
        + "\n".join(rows)
    )
    return get_img("sam1"), get_img("sam2"), get_img("sam3"), summary_md


if ZEROGPU:
    segment_single = spaces.GPU(_segment_single_impl)
    segment_compare = spaces.GPU(_segment_compare_impl)
else:
    segment_single = _segment_single_impl
    segment_compare = _segment_compare_impl


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="SAM Benchmark Sandbox") as demo:
    gr.Markdown(
        "# SAM Benchmark Sandbox\n"
        "Zero-shot text segmentation — SAM1 + CLIP · SAM2 + CLIP · SAM3 (native text)"
    )

    with gr.Tabs():
        with gr.Tab("Single Model"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_in = gr.Image(type="pil", label="Input Image")
                    prompt_in = gr.Textbox(label="Text Prompt", placeholder="e.g. dog, car, person")
                    model_dd = gr.Dropdown(
                        choices=MODEL_NAMES,
                        value="sam3",
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
    demo.launch()
