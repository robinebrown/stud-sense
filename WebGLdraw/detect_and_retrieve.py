#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import torch
import faiss
import numpy as np
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms import Resize
import tensorflow as tf

import config
from build_faiss_index import load_embedder


def load_tflite_detector(model_path: Path):
    """
    Load and return a TFLite interpreter and its I/O details.
    """
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    return interp, inp, out


def detect_and_retrieve(
    img_path: Path,
    det_model: Path,
    embed_model: Path,
    faiss_index: Path,
    faiss_labels: Path,
    device: str,
    det_size: int,
    emb_size: int,
    det_thresh: float
):
    # 1) Load detector
    det_interp, det_inp, det_out = load_tflite_detector(det_model)
    H_det, W_det = det_inp['shape'][1:3]
    det_resize = Resize((H_det, W_det))

    # 2) Load embedder + FAISS
    device_t = torch.device(device)
    model, _ = load_embedder(Path(embed_model), device_t)
    model.eval()
    index = faiss.read_index(str(faiss_index))
    labels = pickle.load(open(faiss_labels, 'rb'))
    emb_resize = Resize((emb_size, emb_size))

    # 3) Read input image
    orig = Image.open(img_path).convert('RGB')
    tensor = read_image(str(img_path)).float() / 255.0

    # 4) Detect
    det_in = det_resize(tensor).numpy().transpose(1,2,0)[None].astype(np.float32)
    det_interp.set_tensor(det_inp['index'], det_in)
    det_interp.invoke()
    dets = det_interp.get_tensor(det_out['index'])[0]

    # 5) Select best detection
    valid = [d for d in dets if d[4] >= det_thresh]
    if valid:
        best = max(valid, key=lambda d: d[4])
    else:
        best = max(dets, key=lambda d: d[4])
    y1, x1, y2, x2, score = best[:5]

    # 6) Crop & embed
    W0, H0 = orig.size
    left, top = int(x1 * W0), int(y1 * H0)
    right, bottom = int(x2 * W0), int(y2 * H0)
    crop = orig.crop((left, top, right, bottom))
    emb_input = emb_resize(torch.from_numpy(np.array(crop)).permute(2,0,1).float() / 255.0)
    with torch.no_grad():
        feat = model(emb_input.unsqueeze(0).to(device_t)).cpu().numpy()

    # 7) FAISS lookup
    _, I = index.search(feat, k=1)
    part_id = labels[I[0][0]]

    print(f"Detected box {(left, top, right, bottom)} -> Part ID: {part_id} (det_conf={score:.2f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-end detection + retrieval demo using TFLite and FAISS.')
    parser.add_argument('--img-path', type=Path, required=True, help='Input image path')
    parser.add_argument('--det-model', type=Path, default=config.YOLO_TFLITE, help='TFLite detector model path')
    parser.add_argument('--embed-model', type=Path, default=config.EMBEDDER_SCRIPTED, help='Embedder model path')
    parser.add_argument('--faiss-index', type=Path, default=config.FAISS_INDEX, help='FAISS index file')
    parser.add_argument('--faiss-labels', type=Path, default=config.FAISS_INDEX.with_suffix('.labels.pkl'), help='FAISS labels pickle file')
    parser.add_argument('--device', type=str, default=config.YOLO_DEVICE, help='Torch device: cpu or cuda')
    parser.add_argument('--det-size', type=int, default=config.YOLO_RUNS, help='Detector input size (square)')
    parser.add_argument('--emb-size', type=int, default=224, help='Embedder input size')
    parser.add_argument('--det-thresh', type=float, default=0.1, help='Min detection confidence')
    args = parser.parse_args()

    detect_and_retrieve(
        img_path=args.img_path,
        det_model=args.det_model,
        embed_model=args.embed_model,
        faiss_index=args.faiss_index,
        faiss_labels=args.faiss_labels,
        device=args.device,
        det_size=args.det_size,
        emb_size=args.emb_size,
        det_thresh=args.det_thresh
    )
