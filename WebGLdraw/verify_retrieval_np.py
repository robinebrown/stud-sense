#!/usr/bin/env python3
import os
import numpy as np
import torch
from build_faiss_index import load_model
from torchvision.io import read_image
from torchvision.transforms import Resize

# Load embedder
model, labels = load_model('embeddings/brick_classifier.pth', torch.device('cpu'))
model.eval()
resize = Resize((224, 224))

# Gather canonical features
features = []
pids = []
for fname in sorted(os.listdir('embeddings/canonical')):
    if not fname.endswith('.png') or '_mask' in fname:
        continue
    pid = fname.split('_')[0]
    img = read_image(os.path.join('embeddings/canonical', fname)).float() / 255.0
    with torch.no_grad():
        feat = model(resize(img).unsqueeze(0)).detach().cpu().numpy()[0]
    features.append(feat)
    pids.append(pid)
features = np.stack(features)  # shape [N, D]

# Embed query image
qimg = read_image('embeddings/canonical/3001_01.png').float() / 255.0
with torch.no_grad():
    qfeat = model(resize(qimg).unsqueeze(0)).detach().cpu().numpy()[0]

# Compute L2 distances, pick nearest
dists = np.sum((features - qfeat)**2, axis=1)
best_idx = int(np.argmin(dists))
print(pids[best_idx])
