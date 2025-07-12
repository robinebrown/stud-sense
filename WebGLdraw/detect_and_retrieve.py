#!/usr/bin/env python3
import argparse, pickle
from pathlib import Path

import torch, faiss, numpy as np
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

from build_faiss_index import load_embedder

def detect_and_retrieve_py(
    img_path, 
    det_pt, 
    embed_pt, 
    faiss_index, 
    faiss_labels, 
    device, 
    det_size, 
    emb_size, 
    det_thresh
):
    # 1) Load YOLO (PyTorch) detector
    yolo = YOLO(det_pt)
    # 2) Load embedder + FAISS
    device_t = torch.device(device)
    embed_model, _ = load_embedder(Path(embed_pt), device_t)
    embed_model.eval()
    index = faiss.read_index(str(faiss_index))
    labels = pickle.load(open(faiss_labels,'rb'))
    # 3) Preprocess for embedder
    emb_pre = transforms.Compose([
        transforms.Resize((emb_size,emb_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    # 4) Run detection (Python API)
    results = yolo.predict(
        source=str(img_path), 
        device=device, 
        imgsz=det_size, 
        conf=det_thresh
    )
    boxes = results[0].boxes.xyxy.cpu().numpy()  # N×4 array
    if len(boxes)==0:
        print("No detections above threshold")
        return
    # Take the highest-confidence box
    box = boxes[0].astype(int)
    x1,y1,x2,y2 = box
    # 5) Crop & embed
    img = Image.open(img_path).convert("RGB")
    crop = img.crop((x1,y1,x2,y2))
    x = emb_pre(crop).unsqueeze(0).to(device_t)
    with torch.no_grad():
        feat = embed_model(x).cpu().numpy()
    # ─── L2-normalize query before FAISS search ───
    feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
    # 6) FAISS lookup
    D, I = index.search(feat, 5)              # get distances & indices of top 5
    print("Top-5 FAISS matches:")
    for dist, idx in zip(D[0], I[0]):
        print(f"  ID {labels[idx]} @ dist {dist:.4f}")
    pid = labels[I[0][0]]                     # still pick the top-1 for the demo
    print(f"Detected box {box.tolist()} → Part ID: {pid}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img-path",     type=Path, required=True)
    p.add_argument("--det-pt",       type=Path, required=True)
    p.add_argument("--embed-pt",     type=Path, required=True)
    p.add_argument("--faiss-index",  type=Path, required=True)
    p.add_argument("--faiss-labels", type=Path, required=True)
    p.add_argument("--device",       type=str, default="cpu")
    p.add_argument("--det-size",     type=int, default=512)
    p.add_argument("--emb-size",     type=int, default=224)
    p.add_argument("--det-thresh",   type=float, default=0.25)
    args = p.parse_args()
    detect_and_retrieve_py(**vars(args))
