#!/usr/bin/env python3
import os
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse

class CanonicalBrickDataset(Dataset):
    def __init__(self, data_dir, size=224):
        self.items = []
        for fname in os.listdir(data_dir):
            if fname.endswith('.png') and '_mask' not in fname:
                pid = fname.split('_')[0]
                path = os.path.join(data_dir, fname)
                self.items.append((path, pid))
        self.labels = sorted({pid for _, pid in self.items})
        self.label_to_idx = {pid: idx for idx, pid in enumerate(self.labels)}
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, pid = self.items[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        label = self.label_to_idx[pid]
        return img, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train an embedding network on canonical brick views.'
    )
    parser.add_argument('--data_dir', required=True, help='Path to canonical views dir')
    parser.add_argument('--output', required=True, help='File to save trained model')
    parser.add_argument('--size', type=int, default=224, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    dataset = CanonicalBrickDataset(args.data_dir, size=args.size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device(args.device)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.labels))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f'Epoch {epoch+1}/{args.epochs} Loss: {running/len(loader):.4f}')

    torch.save({
        'state_dict': model.state_dict(),
        'labels': dataset.labels
    }, args.output)
    print('Saved model to', args.output)
