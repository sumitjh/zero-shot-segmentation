"""Quick end-to-end test: generate SAM masks, rank with CLIP, save overlay."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
from app.models import generate_masks, rank_masks_by_prompt
from app.utils import apply_mask_overlay, draw_bbox

# --- load a test image (download a sample if not present) ---
TEST_IMAGE_PATH = "test_image.jpg"
if not os.path.exists(TEST_IMAGE_PATH):
    import urllib.request
    url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/dog.jpg"
    print(f"Downloading test image...")
    urllib.request.urlretrieve(url, TEST_IMAGE_PATH)

image = np.array(Image.open(TEST_IMAGE_PATH).convert("RGB"))
print(f"Image shape: {image.shape}")

PROMPT = "a dog"
print(f"\nPrompt: '{PROMPT}'")

print("Generating SAM masks...")
masks = generate_masks(image)
print(f"  {len(masks)} masks generated")

print("Ranking with CLIP...")
top_masks = rank_masks_by_prompt(image, masks, PROMPT, top_k=3)

for i, m in enumerate(top_masks):
    print(f"  Rank {i+1}: CLIP score={m['clip_score']:.4f}  area={m['area']}  bbox={[int(v) for v in m['bbox']]}")

# save overlay of top mask
result = apply_mask_overlay(image, top_masks[0])
result = draw_bbox(result, top_masks[0])
Image.fromarray(result).save("test_output.jpg")
print("\nSaved overlay to test_output.jpg")
