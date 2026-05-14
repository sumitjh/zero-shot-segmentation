import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

_sam_instance = None
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"


def get_sam_model() -> SamAutomaticMaskGenerator:
    global _sam_instance
    if _sam_instance is None:
        import os, urllib.request
        checkpoint_path = os.path.expanduser("~/.cache/sam/sam_vit_b_01ec64.pth")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        if not os.path.exists(checkpoint_path):
            print("Downloading SAM ViT-B weights (~375 MB)...")
            urllib.request.urlretrieve(SAM_CHECKPOINT_URL, checkpoint_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=checkpoint_path)
        sam.to(device)
        _sam_instance = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,       # lower = fewer masks, faster
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            min_mask_region_area=500, # ignore tiny regions
        )
    return _sam_instance


def generate_masks(image: np.ndarray) -> list[dict]:
    """Return SAM masks sorted by area descending. Each dict has 'segmentation', 'area', 'bbox'."""
    generator = get_sam_model()
    masks = generator.generate(image)
    masks.sort(key=lambda m: m["area"], reverse=True)
    return masks
