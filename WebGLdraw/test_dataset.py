# test_dataset.py
from synthetic_bricks_v2 import SyntheticBrickDatasetV2
import torchvision.utils as vutils
import os

# ← change this to whichever part you want to smoke-test
part_id = "3823"
obj_path = f"objs/{part_id}.obj"

# Instantiate for just that one .obj
ds = SyntheticBrickDatasetV2(
    obj_dir=obj_path,
    colors_csv="colors.csv",
    elements_csv="elements.csv",
    views_per_obj=5,
    device="cpu"
)

# Prepare output folder
out_dir = f"viz_outputs/smoke/{part_id}"
os.makedirs(out_dir, exist_ok=True)

# Render all samples (here just one) and save
for i in range(len(ds)):
    img, meta = ds[i]
    vutils.save_image(img, f"{out_dir}/sample_{i}.png")
    print(f"{part_id} sample {i} →", meta)
