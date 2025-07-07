#!/usr/bin/env python3
import argparse
from pathlib import Path
import pickle

import torch
import numpy as np
import faiss
from torchvision.transforms import Resize
from torchvision.io import read_image

import config


def load_embedder(checkpoint_path: Path, device: torch.device):
    """
    Load the trained embedder model and return the model and label list (if present).
    """
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    if 'labels' in ckpt:
        labels = ckpt['labels']
        num_labels = len(labels)
    else:
        # Infer from state dict
        num_labels = ckpt['state_dict']['fc.weight'].shape[0]
        labels = None

    model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_labels)
    model.load_state_dict(ckpt['state_dict'])
    # Remove classification head
    model.fc = torch.nn.Identity()
    model.to(device).eval()
    return model, labels


def build_faiss_index(
    embedder: torch.nn.Module,
    canonical_dir: Path,
    device: torch.device
):
    """
    Compute embeddings for all canonical PNGs and build a FAISS IndexIDMap.
    Returns the index and the ordered list of part IDs.
    """
    resize = Resize((config.EMBED_BATCH_SIZE, config.EMBED_BATCH_SIZE))
    items = []
    # Find all canonical images (ignore masks)
    for img_path in canonical_dir.glob("*.png"):
        if img_path.stem.endswith('_mask'):
            continue
        pid = img_path.stem.split('_')[0]
        items.append((img_path, pid))

    part_ids = sorted({pid for _, pid in items})
    label2id = {pid: idx for idx, pid in enumerate(part_ids)}

    features = []
    ids = []
    with torch.no_grad():
        for img_path, pid in items:
            img = read_image(str(img_path)).to(torch.float32) / 255.0
            img = resize(img).unsqueeze(0).to(device)
            feat = embedder(img).cpu().numpy()[0]
            features.append(feat)
            ids.append(label2id[pid])

    features = np.vstack(features).astype('float32')
    ids = np.array(ids, dtype='int64')
    dim = features.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    index.add_with_ids(features, ids)
    return index, part_ids


def main():
    parser = argparse.ArgumentParser(description='Build a FAISS index from canonical renders.')
    parser.add_argument(
        '--embedder', type=Path,
        default=config.EMBEDDER_SCRIPTED,
        help='Path to the scripted embedder (.pt or .pth)'
    )
    parser.add_argument(
        '--canonical-dir', type=Path,
        default=config.CANONICAL_IMAGES_DIR,
        help='Directory of canonical view PNGs'
    )
    parser.add_argument(
        '--output-index', type=Path,
        default=config.FAISS_INDEX,
        help='Output path for FAISS index file'
    )
    parser.add_argument(
        '--output-labels', type=Path,
        default=config.FAISS_INDEX.with_suffix('.labels.pkl'),
        help='Output path for FAISS labels mapping'
    )
    parser.add_argument(
        '--device', type=str,
        default=config.YOLO_DEVICE,
        help='Torch device: cpu or cuda'
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    embedder, labels = load_embedder(args.embedder, device)
    index, part_ids = build_faiss_index(embedder, args.canonical_dir, device)

    # Save index and labels
    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.output_index))
    with open(args.output_labels, 'wb') as f:
        pickle.dump(part_ids, f)

    print(f"Wrote FAISS index to {args.output_index} and labels to {args.output_labels}")


if __name__ == '__main__':
    main()
