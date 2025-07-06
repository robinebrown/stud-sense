# viz_samples.py

import os
import torch
import matplotlib.pyplot as plt
from synthetic_bricks_v2 import SyntheticBrickDatasetV2

# ← change this to point at any .obj you like:
OBJ_PATH   = "objs/3001.obj"
NUM_VIEWS  = 5
IMAGE_SIZE = 330
OUTPUT_DIR = "viz_outputs"

def normalize_img_tensor(tensor):
    """
    Ensure tensor is (3, H, W).  
    Handles cases where channels might be last or there's an extra batch dim.
    """
    # remove singleton batch dim
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    # if it’s H×W×3, move channels first
    if tensor.ndim == 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)
    return tensor

def main():
    if not os.path.isfile(OBJ_PATH):
        raise FileNotFoundError(f"OBJ not found: {OBJ_PATH}")
    print("Rendering mesh:", OBJ_PATH)

    ds = SyntheticBrickDatasetV2(
        obj_dir=OBJ_PATH,
        image_size=IMAGE_SIZE,
        device="cpu",
        views_per_obj=NUM_VIEWS,
    )
    ds.force_gradient = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in range(NUM_VIEWS):
        img_tensor, target = ds[i]   # could be (H,W,3), (3,H,W), or (1,3,H,W)
        mask = target['mask'].cpu().numpy()

        # normalize to (3, H, W)
        img_tensor = normalize_img_tensor(img_tensor)
        assert img_tensor.ndim == 3 and img_tensor.shape[0] == 3, \
            f"After normalization got shape {img_tensor.shape}"

        # to H×W×3 uint8
        rgb = img_tensor.mul(255).byte().permute(1,2,0).cpu().numpy().copy()
        print(f"View {i+1} RGB shape:", rgb.shape)  # should be (256,256,3)

        # save
        rgb_path  = os.path.join(OUTPUT_DIR, f"view_{i+1:02d}_rgb.png")
        mask_path = os.path.join(OUTPUT_DIR, f"view_{i+1:02d}_mask.png")
        plt.imsave(rgb_path, rgb)
        plt.imsave(mask_path, mask, cmap="gray")
        print(f"Saved view {i+1}: {rgb_path}, {mask_path}")

if __name__ == "__main__":
    main()
