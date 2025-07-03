#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
import random
from PIL import Image
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description="Split proto renders named <partID>_<view>.png into YOLO train/val datasets")
    parser.add_argument(
        "--renders-dir", type=str, default="viz_outputs/proto",
        help="Directory containing rendered images <partID>_<view>.png and <partID>_<view>_mask.png")
    parser.add_argument(
        "--output-dir", type=str, default="yolo_dataset",
        help="Root output directory for YOLO-formatted images/labels")
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Fraction of images to use for training (rest go to validation)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling")
    args = parser.parse_args()

    # Gather all rendered images (ignore masks)
    all_imgs = sorted([
        f for f in os.listdir(args.renders_dir)
        if f.endswith('.png') and not f.endswith('_mask.png')
    ])
    random.seed(args.seed)
    random.shuffle(all_imgs)
    n_train = int(len(all_imgs) * args.train_ratio)
    splits = {
        'train': all_imgs[:n_train],
        'val':   all_imgs[n_train:]
    }

    for split, files in splits.items():
        img_out = os.path.join(args.output_dir, 'images', split)
        lbl_out = os.path.join(args.output_dir, 'labels', split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for img_name in files:
            base = os.path.splitext(img_name)[0]  # e.g., '3001_01'
            src_img = os.path.join(args.renders_dir, img_name)
            dst_img = os.path.join(img_out, img_name)
            shutil.copy(src_img, dst_img)

            # Corresponding mask file
            mask_name = f"{base}_mask.png"
            mask_path = os.path.join(args.renders_dir, mask_name)

            if not os.path.exists(mask_path):
                print(f"Warning: missing mask for {img_name}", file=sys.stderr)
                # fall back to full image size
                img_size = Image.open(src_img).size  # (width, height)
                W, H = img_size
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
            label_file = os.path.join(lbl_out, f"{base}.txt")
            with open(label_file, 'w') as f:
                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"Finished splitting into train/val and writing labels under '{args.output_dir}'")

if __name__ == '__main__':
    main()
