#!/usr/bin/env python3
import faiss
import pickle
import torch
from build_faiss_index import load_model
from torchvision.io import read_image
from torchvision.transforms import Resize

index = faiss.read_index('embeddings/faiss_index.bin')
labels = pickle.load(open('embeddings/faiss_index.bin.labels.pkl', 'rb'))
model, _ = load_model('embeddings/brick_classifier.pth', torch.device('cpu'))
img = read_image('embeddings/canonical/3001_01.png').float() / 255.0
feat = model(Resize((224,224))(img).unsqueeze(0)).cpu().numpy()
print(labels[index.search(feat,1)[1][0][0]])
