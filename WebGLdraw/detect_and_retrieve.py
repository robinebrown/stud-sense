#!/usr/bin/env python3
import argparse
from pathlib import Path
import pickle

import torch
import faiss
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms

from build_faiss_index import load_embedder
from train_embed import EmbeddingNet

def detect_and_retrieve_py(
    img_path, det_pt, embed_pt, faiss_index, faiss_labels,
    device='cpu', det_size=512, emb_size=224, det_thresh=0.25
):
    # 1) Load YOLO detector
    yolo = YOLO(det_pt)

    # 2) Run detection
    results = yolo.predict(source=img_path, conf=det_thresh, imgsz=det_size)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # (n,4)
    scores = results[0].boxes.conf.cpu().numpy()  # (n,)
    if len(scores) == 0:
        print("No objects detected above threshold.")
        return
    best = scores.argmax()
    x1, y1, x2, y2 = boxes[best].astype(int)
    print(f"Detected box [{x1}, {y1}, {x2}, {y2}]")

    # 3) Prepare crop for embedding/classification
    img = Image.open(img_path).convert("RGB")
    crop = img.crop((x1, y1, x2, y2))
    device_t = torch.device(device)
    pre = transforms.Compose([
        transforms.Resize((emb_size, emb_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    x = pre(crop).unsqueeze(0).to(device_t)

    # 4) Load embedder for FAISS + build classification model
    embed_model, _ = load_embedder(Path(embed_pt), device_t)
    embed_model.eval()
    labels = pickle.load(open(faiss_labels, 'rb'))
    cls_model = EmbeddingNet(len(labels)).to(device_t)
    cls_model.load_state_dict(torch.load(embed_pt, map_location=device_t))
    cls_model.eval()

    # 5) Classification via ResNet18 head
    with torch.no_grad():
        logits = cls_model(x)                  # (1, N_classes)
        probs  = torch.softmax(logits, dim=1)  # probabilities
        pred_i = int(probs.argmax(dim=1)[0])   # index of highest-prob class
        pred_id = labels[pred_i]               # map back to part ID
    print(f"Classification → Part ID: {pred_id} @ prob {probs[0, pred_i]:.4f}")

    # Optional: FAISS retrieval using embeddings
    # index = faiss.read_index(str(faiss_index))
    # with torch.no_grad():
    #     feat = embed_model(x).cpu().numpy()
    # feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
    # D, I = index.search(feat, 5)
    # print("Top-5 FAISS matches:")
    # for dist, idx in zip(D[0], I[0]):
    #     print(f"  ID {labels[idx]} @ dist {dist:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect and retrieve LEGO part ID.")
    parser.add_argument('--img-path', required=True, help='Path to input image')
    parser.add_argument('--det-pt', required=True, help='Path to TFLite detector model')
    parser.add_argument('--embed-pt', required=True, help='Path to embedder checkpoint (.pt)')
    parser.add_argument('--faiss-index', required=True, help='Path to FAISS index file')
    parser.add_argument('--faiss-labels', required=True, help='Path to FAISS labels pickle')
    parser.add_argument('--device', default='cpu', help='Computation device (cpu/cuda/mps)')
    parser.add_argument('--det-size', type=int, default=512, help='Detector input size')
    parser.add_argument('--emb-size', type=int, default=224, help='Embedder input size')
    parser.add_argument('--det-thresh', type=float, default=0.25, help='Detection confidence threshold')
    args = parser.parse_args()
    detect_and_retrieve_py(
        args.img_path, args.det_pt, args.embed_pt, args.faiss_index,
        args.faiss_labels, args.device, args.det_size, args.emb_size,
        args.det_thresh
    )
