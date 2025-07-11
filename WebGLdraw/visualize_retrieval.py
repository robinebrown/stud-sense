#!/usr/bin/env python3
import os
# on macOS, still allow duplicate OpenMP loads
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
import pickle
import numpy as np
import faiss
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from build_faiss_index import load_embedder

# Paths
BASE          = Path(__file__).parent
test_image    = BASE / 'dataset' / 'lego_yolo' / 'images' / 'test' / '2780_view000.png'
yolo_weights  = BASE / 'runs' / 'detect' / 'train' / 'weights' / 'best.pt'
embedder_pt   = BASE / 'runs' / 'embed_cpu.pt'
index_path    = BASE / 'runs' / 'embed' / 'index.faiss'
labels_path   = BASE / 'runs' / 'embed' / 'index_labels.pkl'
canonical_dir = BASE / 'dataset' / 'canonical'
output_file   = BASE / 'retrieval_visualization.png'

# 1) Detect with Python YOLO API
yolo = YOLO(str(yolo_weights))
res  = yolo.predict(source=str(test_image), device='cpu', imgsz=512, conf=0.25)[0]
boxes = res.boxes.xyxy.cpu().numpy()
if boxes.shape[0] == 0:
    raise RuntimeError("No detection above threshold")
x1,y1,x2,y2 = boxes[0].astype(int)

# 2) Crop test image
orig = Image.open(test_image).convert('RGB')
crop = orig.crop((x1, y1, x2, y2))

# 3) Embed crop
device = torch.device('cpu')
embedder, _ = load_embedder(Path(embedder_pt), device)
embedder.eval()
pre = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
inp = pre(crop).unsqueeze(0).to(device)
with torch.no_grad():
    feat = embedder(inp).cpu().numpy()

# 4) FAISS search
index  = faiss.read_index(str(index_path))
labels = pickle.load(open(labels_path,'rb'))
D, I    = index.search(feat, 3)  # top-3

# 5) Build a composite image
# tile width = crop + 3 canonicals side by side
W, H = crop.size
canvas = Image.new('RGB', (W*4, H+50), (255,255,255))
draw = ImageDraw.Draw(canvas)
# Title text
font = ImageFont.load_default()
draw.text((10,10), f"Test crop (2780):", fill="black", font=font)
canvas.paste(crop, (0,50))
for i, idx in enumerate(I[0]):
    pid = labels[idx]
    dist = D[0,i]
    can = Image.open(canonical_dir/f"{pid}_view0.png").resize((W,H))
    x_off = W*(i+1)
    canvas.paste(can, (x_off,50))
    draw.text((x_off+10,10), f"{pid} (d={dist:.2f})", fill="black", font=font)

# 6) Save
canvas.save(output_file)
print(f"Saved visualization to {output_file}")
