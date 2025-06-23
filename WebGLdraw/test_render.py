# test_render.py
import torch
from synthetic_bricks import SyntheticBrickDataset
import matplotlib.pyplot as plt

# 1) Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2) Load the synthetic dataset
ds = SyntheticBrickDataset(obj_dir="objs", image_size=256)
print("Dataset size:", len(ds))

# 3) Grab one sample
image, target = ds[0]      # image: Tensor (3,H,W), target: dict
mask = target["masks"][0]  # mask: Tensor (H,W)

# 4) Move to CPU and convert to numpy
img_np  = image.permute(1, 2, 0).cpu().numpy()
mask_np = mask.cpu().numpy()

# 5) Plot side by side
plt.figure(figsize=(6,6))

plt.subplot(1,2,1)
plt.title("RGB Render")
plt.imshow(img_np)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Silhouette Mask")
plt.imshow(mask_np, cmap="gray")
plt.axis("off")

plt.show()
