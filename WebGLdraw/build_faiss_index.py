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
    Load the trained embedder model (state dict only or wrapped) and return
    the model (with final fc replaced by Identity) and optional labels list.
    """
    # Load checkpoint, which may be either:
    # - a dict with keys 'state_dict' and optional 'labels'
    # - a plain state_dict mapping param names to tensors
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    # Determine state_dict and labels
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        labels = ckpt.get('labels', None)
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt
        labels = None
    else:
        raise RuntimeError(f"Unrecognized checkpoint format: {checkpoint_path}")
     #── Handle EmbeddingNet’s “backbone.” prefix so keys match torchvision ResNet18 ──#
    if any(k.startswith('backbone.') for k in state_dict.keys()):
     # strip off "backbone." from every key
     state_dict = {
         k[len('backbone.'):]: v
         for k, v in state_dict.items()
     }

    # Infer number of classes (labels) from either labels list or state dict shape
    if labels is not None:
        num_labels = len(labels)
    else:
        # fc.weight shape: [num_classes, feat_dim]
        num_labels = state_dict['fc.weight'].shape[0]

    # Build model and load weights
    model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_labels)
    model.load_state_dict(state_dict)
    # Remove classification head for embedding
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
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
    # ─── L2-normalize each feature vector ───
    norms    = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / norms
    dim = features.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    index.add_with_ids(features, ids)
    return index, part_ids


def main():
    parser = argparse.ArgumentParser(description='Build a FAISS index from canonical renders.')
    parser.add_argument(
        '--embedder', type=Path,
        default=config.EMBEDDER_SCRIPTED,
        help='Path to the embedder checkpoint (state_dict or wrapped)'
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

    args.output_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.output_index))
    with open(args.output_labels, 'wb') as f:
        pickle.dump(part_ids, f)

    print(f"Wrote FAISS index to {args.output_index} and labels to {args.output_labels}")

if __name__ == '__main__':
    main()

