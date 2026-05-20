import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class SAM1Model:
    def __init__(self, checkpoint_path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam.to(device)
        self.generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=8,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            min_mask_region_area=500,
        )
        self.device = device

    def generate_masks(self, image: np.ndarray) -> list[dict]:
        """Return masks sorted by area descending. Each dict has segmentation, area, bbox."""
        masks = self.generator.generate(image)
        masks.sort(key=lambda m: m["area"], reverse=True)
        return masks

    def move_to(self, device: str):
        self.generator.predictor.model.to(device)
        self.device = device
