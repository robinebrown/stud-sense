#!/usr/bin/env python3

import torch
import faiss
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.transforms import Resize, ToTensor

# 1) Load the embedder & FAISS index
from build_faiss_index import load_model
model, _   = load_model("embeddings/brick_classifier.pth", torch.device("cpu"))
model.eval()
index      = faiss.read_index("embeddings/faiss_index.bin")
labels     = pickle.load(open("embeddings/faiss_index.bin.labels.pkl", "rb"))

# Transforms for embedding
resize_emb = Resize((224, 224))
to_tensor  = ToTensor()

# 2) Point to the YOLO-predicted labels folder
labels_dir = Path("runs/detect/predict2/labels")

# 3) Process each .txt file
for txt_file in labels_dir.glob("*.txt"):
    stem = txt_file.stem  # e.g. "3001_01"
    img_path = Path("embeddings/canonical") / f"{stem}.png"
    if not img_path.exists():
        print(f"Image for {stem} not found at {img_path}")
        continue

    # Open the original image
    orig = Image.open(img_path).convert("RGB")
    W, H = orig.size

    # Read each detection line: class_id, x_center, y_center, width, height (all normalized)
    with open(txt_file) as f:
        for line in f:
            cls_id, xc, yc, w, h = map(float, line.split())

            # Only our LEGO class (0) and reasonable box size
            if cls_id != 0.0 or (w * W) * (h * H) < 100:
                continue

            # Convert normalized coords to pixel coords
            x1 = int((xc - w / 2) * W)
            y1 = int((yc - h / 2) * H)
            x2 = int((xc + w / 2) * W)
            y2 = int((yc + h / 2) * H)

            # Crop, resize, and embed
            crop = orig.crop((x1, y1, x2, y2))
            tensor = resize_emb(to_tensor(crop)).unsqueeze(0)
            with torch.no_grad():
                feat = model(tensor).cpu().numpy()

            # FAISS nearest‐neighbor lookup
            _, I = index.search(feat, k=1)
            part_id = labels[I[0][0]]

            print(f"For {stem}: cropped box {(x1, y1, x2, y2)} → Identified part: {part_id}")
