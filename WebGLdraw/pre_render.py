import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from synthetic_bricks import SyntheticBrickDataset

def pre_render(obj_dir, out_dir, views_per_obj=8, image_size=256, max_meshes=None, device="cpu"):
    # 1) Instantiate dataset (CPU or GPU rendering)
    ds = SyntheticBrickDataset(
        obj_dir=obj_dir,
        image_size=image_size,
        max_meshes=max_meshes,
        device=device
    )
    # 2) Wrap for multi-view
    if views_per_obj > 1:
        class MV(torch.utils.data.Dataset):
            def __init__(self, base, v): self.base, self.v = base, v
            def __len__(self): return len(self.base) * self.v
            def __getitem__(self, i): return self.base[i % len(self.base)]
        ds = MV(ds, views_per_obj)

    # 3) Make output directories
    img_dir = os.path.join(out_dir, "images")
    ann_dir = os.path.join(out_dir, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # 4) Loop with progress bar
    for idx in tqdm(range(len(ds)), desc="Rendering dataset", unit="view"):
        image, target = ds[idx]
        # Save RGB image
        arr = (image.mul(255).byte()
                     .permute(1, 2, 0)
                     .cpu().numpy())
        Image.fromarray(arr).save(f"{img_dir}/{idx:06d}.png")
        # Save mask
        mask = target["masks"][0].mul(255).byte().cpu().numpy()
        Image.fromarray(mask).save(f"{img_dir}/{idx:06d}_mask.png")
        # Save annotation (boxes + labels)
        boxes = target["boxes"].cpu()
        labels = target["labels"].cpu()
        torch.save({"boxes": boxes, "labels": labels},
                   f"{ann_dir}/{idx:06d}.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Pre-render synthetic bricks")
    parser.add_argument("--obj_dir",     required=True)
    parser.add_argument("--out_dir",     required=True)
    parser.add_argument("--views_per_obj", type=int, default=8)
    parser.add_argument("--image_size",    type=int, default=256)
    parser.add_argument("--max_meshes",    type=int, default=None)
    parser.add_argument("--device",        default="cpu")
    args = parser.parse_args()
    pre_render(
        obj_dir=args.obj_dir,
        out_dir=args.out_dir,
        views_per_obj=args.views_per_obj,
        image_size=args.image_size,
        max_meshes=args.max_meshes,
        device=args.device
    )
