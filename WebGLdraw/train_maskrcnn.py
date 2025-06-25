# train_maskrcnn.py

import torch.multiprocessing as mp
# Use spawn + file_system sharing to avoid CUDA-fork issues and FD exhaustion
mp.set_start_method('spawn', force=True)
mp.set_sharing_strategy('file_system')

import argparse
import torch
from torch.utils.data import DataLoader
from torch.multiprocessing import get_context
from torchvision.models.detection import maskrcnn_resnet50_fpn
from synthetic_bricks import SyntheticBrickDataset
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from torch.optim.lr_scheduler import MultiStepLR

# create fork context for fast DataLoader workers
fork_ctx = get_context('fork')


class MultiViewDataset(torch.utils.data.Dataset):
    """Repeat each mesh multiple times for multi-view training."""
    def __init__(self, base_ds, views_per_obj=1):
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
    p.add_argument("--obj_dir",       default="objs", help="Folder with .obj meshes")
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch_size",    type=int,   default=10)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--device",        default=None, help="cuda, mps, or cpu")
    p.add_argument("--smoke_test",    action="store_true", help="Use small subset for testing")
    p.add_argument("--views_per_obj", type=int,   default=10, help="Views per mesh per epoch")
    p.add_argument("--image_size",    type=int,   default=256, help="Input height and width")
    p.add_argument("--num_workers",   type=int,   default=16, help="DataLoader worker count")
    return p.parse_args()


def train(obj_dir, epochs, batch_size, lr, device,
          smoke_test, views_per_obj, image_size, num_workers):
    # 1) Load synthetic dataset on CPU (avoid preloading meshes to GPU)
    ds = SyntheticBrickDataset(
        obj_dir=obj_dir,
        image_size=image_size,
        max_meshes=200 if smoke_test else None,
        device="cpu"
    )
    if smoke_test:
        print(f"→ [Smoke-test] using first {len(ds)} meshes")

    # 2) Wrap in MultiViewDataset if needed
    if views_per_obj > 1:
        dataset = MultiViewDataset(ds, views_per_obj)
        print(f"→ Multi-view: {views_per_obj} views per mesh, dataset size {len(dataset)}")
    else:
        dataset = ds

    # 3) DataLoader: fork workers, pin CPU tensors
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        multiprocessing_context=fork_ctx,
        pin_memory=True,
        prefetch_factor=1,
        persistent_workers=False,
    )

    # 4) Model, optimizer, scaler, scheduler
    num_classes = len(ds) + 1  # background + each mesh
    model = maskrcnn_resnet50_fpn(num_classes=num_classes)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler    = GradScaler(device_type="cuda")
    scheduler = MultiStepLR(optimizer, milestones=[8, 12], gamma=0.1)

    # 5) Training loop
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        print(f"\nEpoch {epoch}/{epochs}")
        for i, (images, targets) in enumerate(tqdm(loader, desc="Batches"), 1):
            # move batch to GPU
            images_gpu = [img.to(device, non_blocking=True) for img in images]
            targets_gpu = []
            for t in targets:
                tgt = {
                    k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                    for k, v in t.items()
                }
                targets_gpu.append(tgt)

            # debug: confirm resolution
            if epoch == 1 and i == 1:
                print("image[0].shape =", images_gpu[0].shape)

            # forward + backward
            with autocast():
                loss_dict = model(images_gpu, targets_gpu)
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

        # step scheduler at epoch end
        scheduler.step()

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch} completed. Avg loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"maskrcnn_epoch{epoch}.pth")

    torch.save(model.state_dict(), "maskrcnn_final.pth")
    print("Training complete! Saved maskrcnn_final.pth")


if __name__ == "__main__":
    args = parse_args()
    # select device
    if args.device:
        device = torch.device(args.device)
    else:
        device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    print("Training on device:", device)

    train(
        obj_dir=args.obj_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        smoke_test=args.smoke_test,
        views_per_obj=args.views_per_obj,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
