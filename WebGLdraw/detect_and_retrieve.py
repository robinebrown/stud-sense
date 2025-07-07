import torch
import faiss, pickle
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.io import read_image
from torchvision.transforms import Resize
import tensorflow as tf

# 1) Load TF-Lite detector
det_interp = tf.lite.Interpreter(
    model_path="runs/detect/train/weights/best_saved_model/best_float32.tflite"
)
det_interp.allocate_tensors()
inp_det  = det_interp.get_input_details()[0]
out_det  = det_interp.get_output_details()[0]
H_det, W_det = inp_det["shape"][1:3]
det_resize = Resize((H_det, W_det))

# 2) Load embedder + FAISS
from build_faiss_index import load_model
model, _    = load_model("embeddings/brick_classifier.pth", torch.device("cpu"))
model.eval()
index       = faiss.read_index("embeddings/faiss_index.bin")
labels      = pickle.load(open("embeddings/faiss_index.bin.labels.pkl","rb"))
emb_resize  = Resize((224, 224))

# 3) Read input image
img_path = Path("embeddings/canonical/3001_01.png")
orig     = Image.open(img_path).convert("RGB")
tensor   = read_image(str(img_path)).float() / 255.0

# 4) Detect
det_in = det_resize(tensor).numpy().transpose(1,2,0)[None].astype(np.float32)
det_interp.set_tensor(inp_det["index"], det_in)
det_interp.invoke()
dets = det_interp.get_tensor(out_det["index"])[0]  # shape (N, M)

# 5) Find highest-confidence valid detection
valid = [d for d in dets if d[4] > 0.1]
if not valid:
    # pick the best even if low‐conf
    best = max(dets, key=lambda d: d[4])
    print("  → picked det:", best[:6])

else:
    best = max(valid, key=lambda d: d[4])
y1, x1, y2, x2, score = best[:5]

# 6) Crop & embed
W0, H0 = orig.size
left, top   = int(x1 * W0), int(y1 * H0)
right, bottom = int(x2 * W0), int(y2 * H0)
crop = orig.crop((left, top, right, bottom))
emb_in = emb_resize(torch.from_numpy(np.array(crop)).permute(2,0,1).float() / 255.0)
with torch.no_grad():
    feat = model(emb_in.unsqueeze(0)).cpu().numpy()

# 7) FAISS lookup
_, I = index.search(feat, k=1)
part_id = labels[I[0][0]]
print(f"Detected brick box {left,top,right,bottom}  →  Identified part: {part_id} (det_conf={score:.2f})")
