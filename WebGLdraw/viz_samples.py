# viz_samples.py

import os
import matplotlib.pyplot as plt
import torch
import importlib.util
import sys

# 1) Dynamically load your synthetic_bricks module
spec = importlib.util.spec_from_file_location(
    "synthetic_bricks", os.path.join(os.path.dirname(__file__), "synthetic_bricks.py")
)
synthetic_bricks = importlib.util.module_from_spec(spec)
sys.modules["synthetic_bricks"] = synthetic_bricks
spec.loader.exec_module(synthetic_bricks)

from synthetic_bricks import SyntheticBrickDataset

def main():
    out_dir = "single_brick_viz"
    os.makedirs(out_dir, exist_ok=True)

    # 2) Instantiate the dataset on CPU, pointing at the 3001 OBJ
    ds = SyntheticBrickDataset(
        obj_dir="single_brick/3001.obj",  # path to the 1×1 round solid-stud piece
        image_size=256,
        device="cpu"
    )

    # 3) Render 5 different random views
    for view_num in range(1, 11):
        image, target = ds[0]  # always index 0, but random pose each time
        mask = target["masks"][0].cpu().numpy()

        # clamp RGB to [0,1]
        img_np = image.cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)

        # 4) Save RGB and mask
        plt.imsave(
            os.path.join(out_dir, f"3001_view{view_num}_rgb.png"),
            img_np
        )
        plt.imsave(
            os.path.join(out_dir, f"3001_view{view_num}_mask.png"),
            mask,
            cmap="gray"
        )

    print(f"Saved 5 views of 3001.obj to ./{out_dir}/")

if __name__ == "__main__":
    main()
