#!/usr/bin/env python3
import os, sys
# allow duplicate OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch, faiss, pickle, numpy as np
from pathlib import Path
from PIL import Image

def main(txt_path, img_path):
    print(f"\n=== {txt_path.name} → {img_path.name} ===")

    # 1) Load scripted embedder
    try:
        model = torch.jit.load("embeddings/embedder_scripted.pt", map_location="cpu")
        model.eval()
        print("Loaded TorchScript embedder")
    except Exception as e:
        print("⚠️  Failed to load embedder:", e)
        return

    # 2) Load FAISS index & labels
    try:
        index  = faiss.read_index("embeddings/faiss_index.bin")
        labels = pickle.load(open("embeddings/faiss_index.bin.labels.pkl","rb"))
        print("Loaded FAISS index + labels (", index.ntotal, "entries )")
    except Exception as e:
        print("⚠️  Failed to load FAISS:", e)
        return

    # 3) Load image
    if not img_path.exists():
        print("⚠️  Image not found:", img_path)
        return
    orig = Image.open(img_path).convert("RGB")
    W, H = orig.size
    print(f"Image size: {W}×{H}")

    # 4) Read YOLO .txt
    if not txt_path.exists():
        print("⚠️  Label file not found:", txt_path)
        return
    lines = txt_path.read_text().strip().splitlines()
    print(f"Detected {len(lines)} lines")

    # 5) Process each detection
    for i, line in enumerate(lines, 1):
        cls_id, xc, yc, w, h = map(float, line.split())
        print(f" Line {i}: cls={cls_id}, xc={xc:.4f}, yc={yc:.4f}, w={w:.4f}, h={h:.4f}")

        # filter
        area = (w*W)*(h*H)
        if cls_id != 0.0:
            print("  → skipping (not class 0)")
            continue
        if area < 100:
            print(f"  → skipping (tiny area {area:.1f}px²)")
            continue

        # pixel coords
        x1 = int((xc - w/2)*W); y1 = int((yc - h/2)*H)
        x2 = int((xc + w/2)*W); y2 = int((yc + h/2)*H)
        print(f"  → crop box: ({x1},{y1})→({x2},{y2})")

        # crop & resize
        crop = orig.crop((x1,y1,x2,y2)).resize((224,224), Image.BILINEAR)
        arr  = np.array(crop, np.float32)/255.0    # H×W×3
        tensor = torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0)

        # embed
        try:
            with torch.no_grad():
                feat = model(tensor).cpu().numpy()
            print("  → feature vector shape:", feat.shape)
        except Exception as e:
            print("  ⚠️ embed error:", e)
            continue

        # FAISS lookup
        try:
            D, I = index.search(feat, k=1)
            print("  → FAISS distances:", D, "indices:", I)
            part_id = labels[I[0][0]]
            print("  → Identified part:", part_id)
        except Exception as e:
            print("  ⚠️ FAISS lookup error:", e)

if __name__=="__main__":
    if len(sys.argv)!=3:
        print("Usage: embed_single.py <path/to/label.txt> <path/to/image.png>")
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
