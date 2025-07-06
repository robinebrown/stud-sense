#!/usr/bin/env python3
import os
import torch
import faiss
import pickle
import argparse
import numpy as np
from torchvision import models
from torchvision.transforms import Resize
from torchvision.io import read_image
from PIL import Image


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'labels' in ckpt:
        num_labels = len(ckpt['labels'])
    else:
        num_labels = ckpt['state_dict']['fc.weight'].shape[0]
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_labels)
    model.load_state_dict(ckpt['state_dict'])
    model.fc = torch.nn.Identity()
    return model.to(device).eval(), ckpt.get('labels', None)


def build_index(model, data_dir, device):
    # Resize transform for tensor images
    resize = Resize((224, 224))
    items = []
    for fname in os.listdir(data_dir):
        if not fname.endswith('.png') or '_mask' in fname:
            continue
        pid = fname.split('_')[0]
        items.append((os.path.join(data_dir, fname), pid))
    part_ids = sorted({pid for _, pid in items})
    label2id = {pid: idx for idx, pid in enumerate(part_ids)}

    features = []
    ids = []
    with torch.no_grad():
        for path, pid in items:
            # Load image as Tensor (CxHxW), normalize to [0,1]
            img_tensor = read_image(path).to(torch.float32) / 255.0
            # Resize to model input size
            img_resized = resize(img_tensor)
            tensor = img_resized.unsqueeze(0).to(device)
            feat = model(tensor).cpu().numpy()
            features.append(feat[0])
            ids.append(label2id[pid])

    features = np.vstack(features).astype('float32')
    ids = np.array(ids, dtype='int64')

    # Build FAISS index
    dim = features.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    index.add_with_ids(features, ids)
    return index, part_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a FAISS index from brick embeddings.')
    parser.add_argument('--model',    required=True, help='Path to trained embedder .pth')
    parser.add_argument('--data_dir', required=True, help='Directory of canonical views (.png)')
    parser.add_argument('--output',   required=True, help='Output path for FAISS index file')
    parser.add_argument('--device',   default='cpu', help='Torch device: cpu, mps, or cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    model, labels = load_model(args.model, device)
    index, part_ids = build_index(model, args.data_dir, device)

    # Write out index and labels mapping
    faiss.write_index(index, args.output)
    with open(f"{args.output}.labels.pkl", 'wb') as f:
        pickle.dump(part_ids, f)

    print(f"Wrote FAISS index to {args.output} and labels to {args.output}.labels.pkl")
