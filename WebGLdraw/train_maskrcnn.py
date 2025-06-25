# train_maskrcnn.py

import torch.multiprocessing as mp
# global safety: use spawn + file_system sharing
mp.set_start_method('spawn', force=True)
mp.set_sharing_strategy('file_system')

import argparse
import time
import torch
torch.backends.cudnn.benchmark = True

from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from synthetic_bricks import SyntheticBrickDataset

# we’ll explicitly fork for DataLoader workers
from torch.multiprocessing import get_context
fork_ctx = get_context('fork')

class MultiViewDataset(Dataset):
    """Repeat each mesh N times for multi-view training."""
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
    p = argparse.ArgumentParser("Train Mask R-CNN on synthetic LEGO bricks")
    p.add_argument('--obj_dir',       type=str, required=True)
    p.add_argument('--epochs',        type=int, default=15)
    p.add_argument('--batch_size',    type=int, default=12)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--device',        type=str, default='cuda')
    p.add_argument('--views_per_obj', type=int, default=8)
    p.add_argument('--num_workers',   type=int, default=32)
    p.add_argument('--image_size',    type=int, default=256)
    p.add_argument('--smoke_test',    action='store_true')
    return p.parse_args()

def train(args):
    model_dev = torch.device(args.device)
    cpu_dev   = 'cpu'  # render & load entirely on CPU

    # Build base dataset
    if args.smoke_test:
        print("→ [Smoke-test] loading 200 meshes")
        base_ds = SyntheticBrickDataset(
            obj_dir=args.obj_dir,
            image_size=args.image_size,
            max_meshes=200,
            device=cpu_dev
        )
        views, epochs = 1, 2
    else:
        base_ds = SyntheticBrickDataset(
            obj_dir=args.obj_dir,
            image_size=args.image_size,
            device=cpu_dev
        )
        views, epochs = args.views_per_obj, args.epochs

    ds = MultiViewDataset(base_ds, views)
    print(f"Dataset size: {len(ds)} samples ({len(base_ds)} meshes × {views} views)")

    # DataLoader: fork workers, pin CPU tensors, no persistent_workers, minimal prefetch
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=1,
        collate_fn=collate_fn,
        multiprocessing_context=fork_ctx,
    )

    # Build model
    num_classes = len(base_ds) + 1
    model = maskrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    model.to(model_dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler    = torch.cuda.amp.GradScaler()

    # Training loop (with simple timing)
    for epoch in range(1, epochs+1):
        model.train()
        loss_accum = 0.0

        for i, (imgs_cpu, targs_cpu) in enumerate(loader, 1):
            # measure data→GPU time
            t0 = time.perf_counter()
            images = [img.to(model_dev, non_blocking=True) for img in imgs_cpu]
            targets = []
            for t in targs_cpu:
                d = {}
                for k,v in t.items():
                    d[k] = v.to(model_dev, non_blocking=True) if isinstance(v, torch.Tensor) else v
                targets.append(d)
            data_time = time.perf_counter() - t0

            # GPU forward/backward
            t1 = time.perf_counter()
            with torch.cuda.amp.autocast():
                losses = model(images, targets)
                loss = sum(losses.values())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            gpu_time = time.perf_counter() - t1

            loss_accum += loss.item()
            if i % 50 == 0:
                print(f"Epoch {epoch} Batch {i}/{len(loader)}  "
                      f"data {data_time:.3f}s  gpu {gpu_time:.3f}s  "
                      f"loss {loss_accum/50:.4f}")
                loss_accum = 0.0

        print(f"Epoch {epoch}/{epochs} done, saving maskrcnn_epoch_{epoch}.pth")
        torch.save(model.state_dict(), f"maskrcnn_epoch_{epoch}.pth")

    torch.save(model.state_dict(), "maskrcnn_final.pth")
    print("Training complete — saved maskrcnn_final.pth")

if __name__ == '__main__':
    args = parse_args()
    print(f"Training on {args.device}  |  Rendering on CPU")
    train(args)
