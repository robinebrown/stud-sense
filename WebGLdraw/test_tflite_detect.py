import numpy as np
from pathlib import Path
import tensorflow as tf
from torchvision.io import read_image
from torchvision.transforms import Resize

# Load TFLite model via TF
interpreter = tf.lite.Interpreter(
    model_path="runs/detect/train/weights/best_saved_model/best_float32.tflite"
)
interpreter.allocate_tensors()

# I/O details
input_details  = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
H, W = input_details["shape"][1:3]

# Preprocessor
resize = Resize((H, W))

# Load & preprocess image
img = read_image("embeddings/canonical/3001_01.png").float() / 255.0
img = resize(img).numpy().transpose(1,2,0)[None].astype(np.float32)

# Inference
interpreter.set_tensor(input_details["index"], img)
interpreter.invoke()
dets = interpreter.get_tensor(output_details["index"])  # (1, N, 6)

# Print boxes
for det in dets[0]:
    y1, x1, y2, x2, score, cls = det[:6]
    if score > 0.05:  # lower threshold
        print(f"Detected LEGO at [{x1:.1f},{y1:.1f}]→[{x2:.1f},{y2:.1f}], conf={score:.2f}")

