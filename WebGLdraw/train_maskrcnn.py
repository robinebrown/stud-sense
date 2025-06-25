# train_maskrcnn.py

import multiprocessing as mp
# Use 'spawn' to avoid CUDA re-init errors in forked DataLoader workers
mp.set_start_method('spawn', force=True)

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from synthetic_bricks import SyntheticBrickDataset
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.amp import GradScaler


def collate_fn(batch):
    return tuple(zip(*batch))

class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, ds, views_per_obj):
        self.ds = ds
        self.views_per_obj = views_per_obj

    def __len__(self):
        return len(self.ds) * self.views_per_obj

    def __getitem__(self, idx):
        obj_idx = idx % len(self.ds)
        return self.ds[obj_idx]


def train(obj_dir, epochs, batch_size, lr, device, smoke_test, views_per_obj, num_workers):
    # 1) Load dataset on CPU to avoid per-worker GPU allocations
    ds = SyntheticBrickDataset(
        obj_dir=obj_dir,
        image_size=256,
        max_meshes=200 if smoke_test else None,
        device=torch.device("cpu")
    )
    if smoke_test:
        print(f"→ [Smoke-test] using first {len(ds)} meshes")

    # 2) Multi-view wrapper
    if views_per_obj > 1:
        dataset = MultiViewDataset(ds, views_per_obj)
        print(f"→ Multi-view: {views_per_obj} views per mesh, dataset size {len(dataset)}")
    else:
        dataset = ds

    # 3) High-parallel DataLoader with pin_memory for faster host→device transfers
    #    and persistent_workers disabled to free GPU memory between epochs
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=False
    )

    # 4) Model, optimizer, and AMP scaler
    num_classes = len(ds) + 1  # +1 background
    model = maskrcnn_resnet50_fpn(num_classes=num_classes)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )
    scaler = GradScaler(device.type)

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        running_loss = 0.0
        for i, (images, targets) in enumerate(tqdm(loader, desc="Batches"), 1):
            # Transfer batch from CPU to GPU
            images = [img.to(device) for img in images]
            moved_targets = []
            for t in targets:
                t2 = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                      for k, v in t.items()}
                moved_targets.append(t2)

            with autocast():
                loss_dict = model(images, moved_targets)
                losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += losses.item()
            if i % 50 == 0:
                avg_loss = running_loss / 50
                print(f"Epoch {epoch} Batch {i}/{len(loader)}  loss {avg_loss:.4f}")
                running_loss = 0.0

        torch.save(model.state_dict(), f"maskrcnn_epoch{epoch}.pth")

    torch.save(model.state_dict(), "maskrcnn_final.pth")
    print("Training complete! Model saved to maskrcnn_final.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir", default="objs", help="Folder with .obj meshes")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu")
    parser.add_argument("--smoke_test", action="store_true", help="Use small subset for testing")
    parser.add_argument(
        "--views_per_obj",
        type=int,
        default=10,
        help="Number of random views per mesh per epoch"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="number of DataLoader worker processes"
    )
    args = parser.parse_args()

    # Device auto-selection
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print("Training on device:", device)

    train(
        obj_dir=args.obj_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        smoke_test=args.smoke_test,
        views_per_obj=args.views_per_obj,
        num_workers=args.num_workers
    )
