#!/usr/bin/env python3
import os
import torch
import faiss
import pickle
import argparse
from torchvision import models, transforms
from PIL import Image


def load_model(checkpoint_path, device):
    # Load checkpoint and rebuild model
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_labels = len(ckpt['labels']) if 'labels' in ckpt else ckpt['state_dict']['fc.weight'].shape[0]
    model = models.resnet18(pretrained=False)
    # original fc replaced in training, now rebuild for loading
    model.fc = torch.nn.Linear(model.fc.in_features, num_labels)
    model.load_state_dict(ckpt['state_dict'])
    # replace fc with identity to extract features
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    return model, ckpt.get('labels', None)


def build_index(model, data_dir, device):
    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Gather all image paths
    items = []  # list of (path, pid)
    for fname in os.listdir(data_dir):
        if not fname.endswith('.png') or '_mask' in fname:
            continue
        pid = fname.split('_')[0]
        items.append((os.path.join(data_dir, fname), pid))
    # Unique part IDs
    part_ids = sorted({pid for _, pid in items})
    label2id = {pid: idx for idx, pid in enumerate(part_ids)}

    # Extract features
    features = []
    ids = []
    with torch.no_grad():
        for path, pid in items:
            img = Image.open(path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
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
    parser = argparse.ArgumentParser(
        description='Build a FAISS index from canonical brick embeddings.'
    )
    parser.add_argument('--model', required=True,
                        help='Path to trained embedder .pth checkpoint')
    parser.add_argument('--data_dir', required=True,
                        help='Directory of canonical views (.png images)')
    parser.add_argument('--output', required=True,
                        help='Output path for FAISS index file')
    parser.add_argument('--device', default='cpu',
                        help='Torch device: cpu, mps, or cuda')
    args = parser.parse_args()

    device = torch.device(args.device)
    model, labels = load_model(args.model, device)
    index, part_ids = build_index(model, args.data_dir, device)

    # Save FAISS index
    faiss.write_index(index, args.output)
    # Save mapping from ID to part label
    with open(args.output + '.labels.pkl', 'wb') as f:
        pickle.dump(part_ids, f)

    print(f'Wrote FAISS index to {args.output} and labels map to {args.output}.labels.pkl')
