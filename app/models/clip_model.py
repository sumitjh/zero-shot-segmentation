import numpy as np
import torch
import clip
from PIL import Image

_clip_model = None
_clip_preprocess = None


def get_clip_model():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
    return _clip_model, _clip_preprocess


def rank_masks_by_prompt(
    image: np.ndarray,
    masks: list[dict],
    prompt: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Score each SAM mask region against the text prompt using CLIP.
    Returns top_k masks sorted by CLIP similarity descending,
    each with an added 'clip_score' field.
    """
    model, preprocess = get_clip_model()
    device = next(model.parameters()).device

    text = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    scored = []
    pil_image = Image.fromarray(image)

    for mask in masks:
        x, y, w, h = mask["bbox"]  # xywh
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w < 10 or h < 10:
            continue
        # Apply mask to full image: zero out non-mask pixels, then crop to bbox.
        # This gives CLIP spatial context and penalises partial/fragment regions.
        masked = image.copy()
        masked[~mask["segmentation"]] = 0
        crop = Image.fromarray(masked).crop((x, y, x + w, y + h))
        tensor = preprocess(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            img_features = model.encode_image(tensor)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            score = (img_features @ text_features.T).item()
        scored.append({**mask, "clip_score": score})

    scored.sort(key=lambda m: m["clip_score"], reverse=True)
    return scored[:top_k]
