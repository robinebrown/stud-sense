#!/usr/bin/env python3
import os
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
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
            # augmentations for robustness
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, pid = self.items[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        label = self.label_to_idx[pid]
        return img, label

class EmbeddingNet(nn.Module):
    """
    A simple embedding network based on ResNet18. Outputs a vector of dimension equal
    to the number of canonical brick classes.
    """
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train an embedding network on canonical brick views.'
    )
    parser.add_argument('--data_dir', required=True, help='Path to canonical views dir')
    parser.add_argument('--output', required=True, help='File to save trained model state_dict')
    parser.add_argument('--size', type=int, default=224, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # Prepare dataset and dataloader
    dataset = CanonicalBrickDataset(args.data_dir, size=args.size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device(args.device)

    # Instantiate embedding network
    model = EmbeddingNet(len(dataset.labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        print(f'Epoch {epoch+1}/{args.epochs} Loss: {avg_loss:.4f}')

    # Save the trained weights and class labels
    torch.save(model.state_dict(), args.output)
    labels_path = os.path.splitext(args.output)[0] + '_labels.pkl'
    with open(labels_path, 'wb') as f:
        torch.save(dataset.labels, f)
    print('Saved model state_dict to', args.output)
    print('Saved class labels to', labels_path)
