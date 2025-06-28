# test_dataset.py
import os
import torchvision.utils as vutils
from synthetic_bricks_v2 import SyntheticBrickDatasetV2

# ← change this to whichever part you want to smoke-test
part_id = "3006"
obj_path = f"objs/{part_id}.obj"

# Instantiate for just that one .obj
ds = SyntheticBrickDatasetV2(
    obj_dir=obj_path,
    views_per_obj=5,
    device="cpu"
)

# Prepare output folder
out_dir = f"viz_outputs/smoke/{part_id}"
os.makedirs(out_dir, exist_ok=True)

# Render all samples and save both image + mask
for i in range(len(ds)):
    img, meta = ds[i]                   # img: [3,H,W], meta['mask']: [H,W]
    mask = meta['mask'].unsqueeze(0)    # [1,H,W], ready for save_image

    # RGB render
    vutils.save_image(img, f"{out_dir}/sample_{i:02d}.png")
    # Mask (0/1 → black/white)
    vutils.save_image(mask.float(), f"{out_dir}/sample_{i:02d}_mask.png")

    print(f"{part_id} sample {i} →", meta)
