import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class SAM2Model:
    def __init__(self, checkpoint_path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_sam2("sam2_hiera_b+.yaml", checkpoint_path, device=device)
        self.generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=8,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            min_mask_region_area=500,
        )
        self.device = device

    def generate_masks(self, image: np.ndarray) -> list[dict]:
        """Return masks sorted by area descending. Same format as SAM1."""
        masks = self.generator.generate(image)
        masks.sort(key=lambda m: m["area"], reverse=True)
        return masks

    def move_to(self, device: str):
        self.generator.predictor.model.to(device)
        self.device = device
