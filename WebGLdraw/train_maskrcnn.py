# train_maskrcnn.py

import torch.multiprocessing as mp
# spawn + file_system for safe CUDA+multiprocessing
mp.set_start_method('spawn', force=True)
mp.set_sharing_strategy('file_system')

import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torch.multiprocessing import get_context
from torchvision.models.detection import maskrcnn_resnet50_fpn
from synthetic_bricks import SyntheticBrickDataset
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from torch.optim.lr_scheduler import MultiStepLR

spawn_ctx = get_context('spawn')

class MultiViewDataset(Dataset):
    def __init__(self, base_dataset, views_per_obj=1):
        self.base = base_dataset
        self.views = views_per_obj

    def __len__(self):
        return len(self.base) * self.views

    def __getitem__(self, idx):
        return self.base[idx % len(self.base)]

def collate_fn(batch):
    imgs, tars = zip(*batch)
    return list(imgs), list(tars)

def parse_args():
    p = argparse.ArgumentParser("Train Mask R-CNN on synthetic LEGO bricks")
    p.add_argument("--obj_dir",       default="objs")
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch_size",    type=int,   default=10)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--device",        default=None)
    p.add_argument("--smoke_test",    action="store_true")
    p.add_argument("--views_per_obj", type=int,   default=8)
    p.add_argument("--image_size",    type=int,   default=256)
    p.add_argument("--num_workers",   type=int,   default=16)
    return p.parse_args()

def train(args):
    # 1) Render on GPU, but outputs are moved to CPU in the Dataset
    ds = SyntheticBrickDataset(
        obj_dir=args.obj_dir,
        image_size=args.image_size,
        max_meshes=200 if args.smoke_test else None,
        device=args.device
    )
    if args.smoke_test:
        print(f"→ [Smoke-test] using first {len(ds)} meshes")

    # 2) Multi-view wrapper
    if args.views_per_obj > 1:
        dataset = MultiViewDataset(ds, args.views_per_obj)
        print(f"→ Multi-view: {args.views_per_obj}×, size={len(dataset)}")
    else:
        dataset = ds

    # 3) Fast DataLoader with CUDA-capable workers
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        multiprocessing_context=spawn_ctx,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=collate_fn
    )

    # 4) Model, optimizer, scaler, scheduler
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = maskrcnn_resnet50_fpn(num_classes=len(ds) + 1).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler    = GradScaler()
    scheduler = MultiStepLR(optimizer, milestones=[8, 12], gamma=0.1)

    # 5) Training loop
    for e in range(1, args.epochs + 1):
        print(f"\nEpoch {e}/{args.epochs}")
        run_loss = 0.0

        for i, (imgs, tars) in enumerate(tqdm(loader, desc="Batches"), 1):
            # GPU training batch
            imgs_gpu = [im.to(device, non_blocking=True) for im in imgs]
            tars_gpu = [{k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                         for k,v in tar.items()} for tar in tars]

            if e == 1 and i == 1:
                print("image[0].shape =", imgs_gpu[0].shape)

            with autocast():
                losses = model(imgs_gpu, tars_gpu)
                loss = sum(losses.values())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item()
            if i % 50 == 0:
                print(f"  Batch {i}/{len(loader)}  Loss: {run_loss/50:.4f}")
                run_loss = 0.0

        scheduler.step()
        torch.save(model.state_dict(), f"maskrcnn_epoch{e}.pth")

    torch.save(model.state_dict(), "maskrcnn_final.pth")
    print("Training complete — maskrcnn_final.pth saved")


if __name__ == "__main__":
    args = parse_args()
    print("Training on device:", args.device or "cuda")
    train(args)
