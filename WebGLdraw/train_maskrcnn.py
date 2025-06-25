# train_maskrcnn.py

import torch.multiprocessing as mp
# Use safe spawn start method
mp.set_start_method('spawn', force=True)

import argparse
import torch
# Speed up fixed-size conv workloads
torch.backends.cudnn.benchmark = True

from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from synthetic_bricks import SyntheticBrickDataset

class MultiViewDataset(Dataset):
    """Repeats each mesh N times for multi-view training."""
    def __init__(self, base_dataset, views_per_obj):
        self.base = base_dataset
        self.views = views_per_obj

    def __len__(self):
        return len(self.base) * self.views

    def __getitem__(self, idx):
        return self.base[idx % len(self.base)]

def collate_fn(batch):
    return tuple(zip(*batch))

def parse_args():
    p = argparse.ArgumentParser(description="Train Mask R-CNN on synthetic LEGO bricks")
    p.add_argument('--obj_dir',       type=str, required=True,
                   help='directory containing .obj files')
    p.add_argument('--epochs',        type=int, default=15)
    p.add_argument('--batch_size',    type=int, default=12)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--device',        type=str, default='cuda',
                   help='device for model & training (e.g. cuda or cpu)')
    p.add_argument('--views_per_obj', type=int, default=8)
    p.add_argument('--num_workers',   type=int, default=32)
    p.add_argument('--image_size',    type=int, default=256)
    p.add_argument('--smoke_test',    action='store_true',
                   help='quick smoke-test on 200 meshes, 2 epochs, 1 view')
    return p.parse_args()

def train(args):
    # Model / train device
    model_dev = torch.device(args.device)

    # Always render & load dataset on CPU
    ds_dev = 'cpu'

    # Build the base dataset
    if args.smoke_test:
        print("→ [Smoke-test] using first 200 meshes")
        base_ds = SyntheticBrickDataset(
            obj_dir=args.obj_dir,
            image_size=args.image_size,
            max_meshes=200,
            device=ds_dev
        )
        views, epochs = 1, 2
    else:
        base_ds = SyntheticBrickDataset(
            obj_dir=args.obj_dir,
            image_size=args.image_size,
            device=ds_dev
        )
        views, epochs = args.views_per_obj, args.epochs

    ds = MultiViewDataset(base_ds, views)
    print(f"Dataset size: {len(ds)} samples ({len(base_ds)} meshes × {views} views)")

    # DataLoader: pin_memory=False because data are CPU tensors
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )

    # Build Mask R-CNN
    num_classes = len(base_ds) + 1  # background + each mesh class
    model = maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    model.to(model_dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler    = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(1, epochs+1):
        model.train()
        loss_accum = 0.0

        for i, (images_cpu, targets_cpu) in enumerate(loader, 1):
            # Move only Tensor fields onto the GPU
            images  = [img.to(model_dev, non_blocking=True) for img in images_cpu]
            targets = []
            for t in targets_cpu:
                t2 = {}
                for k, v in t.items():
                    # only torch.Tensor gets moved
                    t2[k] = v.to(model_dev, non_blocking=True) if isinstance(v, torch.Tensor) else v
                targets.append(t2)

            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_accum += loss.item()
            if i % 50 == 0:
                print(f"Batch {i}/{len(loader)}  avg loss: {loss_accum/50:.4f}")
                loss_accum = 0.0

        print(f"Epoch {epoch}/{epochs} done — saving maskrcnn_epoch_{epoch}.pth")
        torch.save(model.state_dict(), f"maskrcnn_epoch_{epoch}.pth")

    torch.save(model.state_dict(), "maskrcnn_final.pth")
    print("Training complete. Final model saved to maskrcnn_final.pth")

if __name__ == '__main__':
    args = parse_args()
    print(f"Training on device: {args.device}  |  Rendering on CPU")
    train(args)
