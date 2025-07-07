#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np

import config


def split_proto_renders(renders_dir: Path, output_dir: Path, train_ratio: float, seed: int):
    """
    Split prototype renders into YOLO train/validation datasets.
    Each image <partID>_<view>.png must have a corresponding mask <partID>_<view>_mask.png.
    """
    # Gather all rendered images (ignore masks)
    all_imgs = sorted([
        f for f in os.listdir(renders_dir)
        if f.endswith('.png') and not f.endswith('_mask.png')
    ])

    random.seed(seed)
    random.shuffle(all_imgs)
    n_train = int(len(all_imgs) * train_ratio)
    splits = {
        'train': all_imgs[:n_train],
        'val':   all_imgs[n_train:]
    }

    for split, files in splits.items():
        img_out = output_dir / 'images' / split
        lbl_out = output_dir / 'labels' / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_name in files:
            base = Path(img_name).stem
            src_img = renders_dir / img_name
            dst_img = img_out / img_name
            shutil.copy(src_img, dst_img)

            mask_name = f"{base}_mask.png"
            mask_path = renders_dir / mask_name

            # Compute bounding box
            if not mask_path.exists():
                # Fall back to full image
                W, H = Image.open(src_img).size
                xc, yc, bw, bh = 0.5, 0.5, 1.0, 1.0
            else:
                mask = np.array(Image.open(mask_path).convert('L'))
                ys, xs = np.where(mask > 0)
                H, W = mask.shape
                if ys.size:
                    y0, y1 = ys.min(), ys.max()
                    x0, x1 = xs.min(), xs.max()
                    bw = (x1 - x0 + 1) / W
                    bh = (y1 - y0 + 1) / H
                    xc = (x0 + (x1 - x0 + 1) / 2) / W
                    yc = (y0 + (y1 - y0 + 1) / 2) / H
                else:
                    xc, yc, bw, bh = 0.5, 0.5, 1.0, 1.0

            # Write YOLO label (class 0)
            label_file = lbl_out / f"{base}.txt"
            with open(label_file, 'w') as f:
                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"Finished splitting into train/val under '{output_dir}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Split proto renders into YOLO train/validation datasets"
    )
    default_renders = config.BASE_DIR / 'viz_outputs' / 'proto'
    default_output = config.BASE_DIR / 'yolo_dataset'

    parser.add_argument(
        '--renders-dir', type=Path,
        default=default_renders,
        help='Directory containing rendered PNGs and masks'
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=default_output,
        help='Root YOLO output directory'
    )
    parser.add_argument(
        '--train-ratio', type=float,
        default=0.8,
        help='Fraction of images to use for training'
    )
    parser.add_argument(
        '--seed', type=int,
        default=42,
        help='Random seed for shuffling'
    )
    args = parser.parse_args()

    split_proto_renders(
        renders_dir=args.renders_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
