---
title: SAM Benchmark Sandbox
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
hardware: zero-gpu
---

# SAM Benchmark Sandbox

Zero-shot text segmentation with three SAM variants — pick a model or run all three side-by-side.

## Models

| Model | Architecture | Text grounding | VRAM (local RTX 2060S) |
|-------|-------------|----------------|------------------------|
| **SAM1 + CLIP** | ViT-B automatic mask generator | Post-hoc CLIP ranking | ~3.1 GB |
| **SAM2 + CLIP** | Hiera-B+ automatic mask generator | Post-hoc CLIP ranking | ~2.9 GB |
| **SAM3.1** | ViT + VL backbone | Native text prompting | ~4.6 GB |

## Benchmark (Yosemite, prompt: "mountain", top-3)

| Model | Inference | Peak VRAM | Semantic accuracy |
|-------|-----------|-----------|-------------------|
| SAM1 + CLIP | 1128 ms | 3121 MB | Coarse — finds large regions near mountains |
| SAM2 + CLIP | 910 ms | 2881 MB | Better mask quality, still CLIP-ranked |
| SAM3.1 | 1242 ms | 4588 MB | Best — directly finds rock faces by text prompt |

## How it works

**SAM1 / SAM2 + CLIP**: generate all candidate masks automatically (no text), then rank them against the text prompt using CLIP similarity. The top-K masks are returned. Segmentation quality depends on SAM's automatic generator, and semantic accuracy depends on CLIP's zero-shot scoring.

**SAM3.1**: uses a native visual-language backbone (ViT + text encoder fused early). The model directly outputs masks conditioned on the text prompt — no post-hoc re-ranking needed. Produces more semantically precise results, especially for specific object categories.

## Run locally

```bash
conda activate cv-sam3
python -m app.gradio_app          # local Gradio UI on port 7860
uvicorn app.main:app --port 8000  # FastAPI + Swagger UI
```

## Tech stack

- PyTorch · SAM (Meta) · SAM2 (Meta) · SAM3.1 (Meta) · OpenCLIP · Gradio · FastAPI
