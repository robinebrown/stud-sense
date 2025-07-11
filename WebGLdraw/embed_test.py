# embed_test.py

import pickle
import torch
import faiss
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from train_embed import EmbeddingNet   # ← your embedder class

# 1) Load FAISS index & labels
index = faiss.read_index("runs/embed11/index.faiss")
labels = pickle.load(open("runs/embed11/index_labels.pkl", "rb"))

# 2) Load your trained embedder
model = EmbeddingNet()
ckpt = torch.load("runs/embed11/weights/best.pt", map_location="cpu")
model.load_state_dict(ckpt)
model.eval()

# 3) Pre-processing (must match training)
preproc = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225]),
])

# 4) Load & crop the YOLO test image
box = (48, 72, 462, 416)  # coordinates from your detection
img_path = Path("dataset/lego_yolo/images/test/3001_11.png")
img = Image.open(img_path).convert("RGB").crop(box)

# 5) Embed & query FAISS
x = preproc(img).unsqueeze(0)           # shape (1,3,224,224)
with torch.no_grad():
    feat = model(x).cpu().numpy()      # (1, D)

D, I = index.search(feat, 5)            # top-5
print("Top-5 distances:", np.round(D[0],4))
print("Top-5 part IDs: ", [labels[i] for i in I[0]])
