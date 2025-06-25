# train_maskrcnn.py

import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from synthetic_bricks import SyntheticBrickDataset
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from torch.optim.lr_scheduler import MultiStepLR

class MultiViewDataset(Dataset):
    """Repeat each mesh multiple times for multi‐view training."""
    def __init__(self, base_ds, views_per_obj):
        self.base = base_ds
        self.views = views_per_obj

    def __len__(self):
        return len(self.base) * self.views

    def __getitem__(self, idx):
        return self.base[idx % len(self.base)]

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def parse_args():
    p = argparse.ArgumentParser("Train Mask R-CNN on synthetic LEGO bricks")
    p.add_argument("--obj_dir",       type=str,   required=True,
                   help="Path to folder of .obj meshes")
    p.add_argument("--epochs",        type=int,   default=15)
    p.add_argument("--batch_size",    type=int,   default=20)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--device",        type=str,   default="cuda",
                   help="‘cuda’ or ‘cpu’ for training")
    p.add_argument("--smoke_test",    action="store_true",
                   help="Quick test on 200 meshes")
    p.add_argument("--views_per_obj", type=int,   default=8,
                   help="Random views per mesh/epoch")
    p.add_argument("--image_size",    type=int,   default=256,
                   help="Input resolution (H and W)")
    p.add_argument("--num_workers",   type=int,   default=16,
                   help="DataLoader worker count")
    return p.parse_args()

def train(args):
    # 1) Render on CPU, training on GPU
    ds = SyntheticBrickDataset(
        obj_dir=args.obj_dir,
        image_size=args.image_size,
        max_meshes=200 if args.smoke_test else None,
        device="cpu"   # <-- CPU rendering
    )
    if args.smoke_test:
        print(f"→ [Smoke-test] using first {len(ds)} meshes")

    # 2) Multi‐view wrapper
    if args.views_per_obj > 1:
        dataset = MultiViewDataset(ds, args.views_per_obj)
        print(f"→ Multi-view: {args.views_per_obj}×, total samples = {len(dataset)}")
    else:
        dataset = ds

    # 3) DataLoader: CPU workers generate & pin batches
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=collate_fn
    )

    # 4) Model, optimizer, AMP scaler, scheduler
    device = torch.device(args.device if args.device else "cuda")
    model = maskrcnn_resnet50_fpn(num_classes=len(ds)+1).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler    = GradScaler()
    scheduler = MultiStepLR(optimizer, milestones=[8,12], gamma=0.1)

    # 5) Training loop
    for epoch in range(1, args.epochs+1):
        running_loss = 0.0
        print(f"\nEpoch {epoch}/{args.epochs}")
        for i, (images, targets) in enumerate(tqdm(loader, desc="Batches"), 1):
            # Move CPU‐rendered batch onto GPU
            imgs_gpu = [img.to(device, non_blocking=True) for img in images]
            targs_gpu = []
            for tar in targets:
                tg = {
                    k: (v.to(device, non_blocking=True)
                        if isinstance(v, torch.Tensor) else v)
                    for k,v in tar.items()
                }
                targs_gpu.append(tg)

            # Debug: confirm shape once
            if epoch==1 and i==1:
                print("image[0].shape =", imgs_gpu[0].shape)

            with autocast():
                loss_dict = model(imgs_gpu, targs_gpu)
                loss = sum(loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if i % 50 == 0:
                avg = running_loss / 50
                print(f"  Batch {i}/{len(loader)}  Loss: {avg:.4f}")
                running_loss = 0.0

        scheduler.step()
        torch.save(model.state_dict(), f"maskrcnn_epoch{epoch}.pth")

    torch.save(model.state_dict(), "maskrcnn_final.pth")
    print("Training complete! Model saved to maskrcnn_final.pth")

if __name__=="__main__":
    args = parse_args()
    print("Training on device:", args.device)
    train(args)
