#!/usr/bin/env python3
import argparse
from pathlib import Path
import pickle

import torch
import faiss
from PIL import Image
from torchvision.transforms import Resize, ToTensor

import config
from build_faiss_index import load_model

def crop_and_embed(
    labels_dir: Path,
    canonical_dir: Path,
    embedder_path: Path,
    index_path: Path,
    labels_file: Path,
    device: str,
    img_size: int,
    min_box_area: int
):
    """
    Crop detected boxes from YOLO outputs, embed with a model, and retrieve part IDs via FAISS.
    """
    # Load embedder and FAISS index
    model, _ = load_model(str(embedder_path), torch.device(device))
    model.eval()
    index = faiss.read_index(str(index_path))
    with open(labels_file, 'rb') as f:
        idx2label = pickle.load(f)

    # Transforms for embedding
    resize = Resize((img_size, img_size))
    to_tensor = ToTensor()

    # Process each YOLO label file
    for txt_file in labels_dir.glob("*.txt"):
        stem = txt_file.stem  # e.g. "3001_01"
        img_path = canonical_dir / f"{stem}.png"
        if not img_path.exists():
            print(f"Image for {stem} not found at {img_path}")
            continue

        orig = Image.open(img_path).convert("RGB")
        W, H = orig.size

        with open(txt_file) as f:
            for line in f:
                cls_id, xc, yc, w, h = map(float, line.split())
                # Only our LEGO class (0) and reasonable box size
                if cls_id != 0.0 or (w * W) * (h * H) < min_box_area:
                    continue

                # Convert normalized coords to pixel coords
                x1 = int((xc - w / 2) * W)
                y1 = int((yc - h / 2) * H)
                x2 = int((xc + w / 2) * W)
                y2 = int((yc + h / 2) * H)

                # Crop, resize, and embed
                crop = orig.crop((x1, y1, x2, y2))
                tensor = to_tensor(resize(crop)).unsqueeze(0)
                with torch.no_grad():
                    feat = model(tensor).cpu().numpy()

                # FAISS nearest-neighbor lookup
                _, I = index.search(feat, k=1)
                part_id = idx2label[I[0][0]]

                print(f"For {stem}: cropped box {(x1, y1, x2, y2)} → Identified part: {part_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop YOLO detections, embed, and map to part IDs using FAISS."
    )
    parser.add_argument(
        "--labels-dir", type=Path,
        default=Path("runs/detect/predict2/labels"),
        help="Directory of YOLO detection label files"
    )
    parser.add_argument(
        "--canonical-dir", type=Path,
        default=config.CANONICAL_IMAGES_DIR,
        help="Directory of canonical images for embedding"
    )
    parser.add_argument(
        "--embedder-model", type=Path,
        default=config.EMBEDDINGS_DIR / "brick_classifier.pth",
        help="Path to the embedder model"
    )
    parser.add_argument(
        "--index", type=Path,
        default=config.EMBEDDINGS_DIR / "faiss_index.bin",
        help="Path to the FAISS index file"
    )
    parser.add_argument(
        "--labels-file", type=Path,
        default=config.EMBEDDINGS_DIR / "faiss_index.bin.labels.pkl",
        help="Path to the FAISS labels pickle file"
    )
    parser.add_argument(
        "--device", type=str,
        default="cpu",
        help="Torch device (cpu or cuda)"
    )
    parser.add_argument(
        "--img-size", type=int,
        default=224,
        help="Image size for embedder input"
    )
    parser.add_argument(
        "--min-box-area", type=int,
        default=100,
        help="Minimum bounding box area (pixels) to consider"
    )
    args = parser.parse_args()

    crop_and_embed(
        labels_dir=args.labels_dir,
        canonical_dir=args.canonical_dir,
        embedder_path=args.embedder_model,
        index_path=args.index,
        labels_file=args.labels_file,
        device=args.device,
        img_size=args.img_size,
        min_box_area=args.min_box_area
    )
