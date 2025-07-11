# viz_crop.py
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

img = Image.open("dataset/lego_yolo/images/test/3001_11.png")
draw = ImageDraw.Draw(img)
box = (48, 72, 462, 416)
draw.rectangle(box, outline="red", width=3)

plt.imshow(img)
plt.axis("off")
plt.show()