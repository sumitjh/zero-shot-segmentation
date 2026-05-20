import numpy as np
from PIL import Image


def apply_mask_overlay(image: np.ndarray, mask: dict, color: tuple = (0, 120, 255), alpha: float = 0.45) -> np.ndarray:
    """Blend a colored mask overlay onto the image."""
    output = image.copy().astype(np.float32)
    seg = mask["segmentation"]  # bool HxW
    for c, v in enumerate(color):
        output[:, :, c] = np.where(seg, output[:, :, c] * (1 - alpha) + v * alpha, output[:, :, c])
    return np.clip(output, 0, 255).astype(np.uint8)


def draw_bbox(image: np.ndarray, mask: dict, color: tuple = (0, 120, 255), thickness: int = 2) -> np.ndarray:
    import cv2
    x, y, w, h = [int(v) for v in mask["bbox"]]
    return cv2.rectangle(image.copy(), (x, y), (x + w, y + h), color, thickness)


def masks_to_response(masks: list[dict], include_rle: bool = False) -> list[dict]:
    """Serialize masks for API response (strip numpy arrays, keep metadata)."""
    out = []
    for i, m in enumerate(masks):
        entry = {
            "rank": i + 1,
            "clip_score": round(m.get("clip_score", 0.0), 4),
            "area": int(m["area"]),
            "bbox": [int(v) for v in m["bbox"]],
        }
        if include_rle:
            seg = m["segmentation"]
            entry["mask_rle"] = seg.flatten().tolist()
        out.append(entry)
    return out
