# train_maskrcnn.py

import argparse
import glob
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from synthetic_bricks import SyntheticBrickDataset
from tqdm import tqdm

# ─── Ensure CUDA works in subprocesses ───────────────────────────────────────────
if torch.cuda.is_available():
    mp.set_start_method('spawn', force=True)

class MultiViewDataset:
    """Repeat each object multiple times for multiple random views."""
    def __init__(self, base_ds, views_per_obj=1):
        self.base  = base_ds
        self.views = views_per_obj

    def __len__(self):
        return len(self.base) * self.views

    def __getitem__(self, idx):
        obj_idx = idx % len(self.base)
        return self.base[obj_idx]

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def train(obj_dir, epochs, batch_size, lr, device,
          smoke_test=False, views_per_obj=1):
    # 1) Gather all .obj paths
    all_objs = sorted(glob.glob(f"{obj_dir}/*.obj"))
    print(f"→ Found {len(all_objs)} .obj files in {obj_dir}")
    if smoke_test:
        all_objs = all_objs[:200]
        print(f"→ [Smoke-test] Using {len(all_objs)} files")

    # 2) Build dataset
    base_ds = SyntheticBrickDataset(
        obj_dir=obj_dir,
        image_size=256,
        max_tries=5
    )

    # 3) Optionally wrap for multi-view
    if views_per_obj > 1:
        ds = MultiViewDataset(base_ds, views_per_obj)
        print(f"→ Multi-view: {views_per_obj}× → dataset size {len(ds)}")
    else:
        ds = base_ds

    # 4) DataLoader (single-process to avoid CUDA fork issues)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,       # ← single-process
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 5) Model & optimizer
    num_classes = len(base_ds) + 1  # +1 for background
    model = maskrcnn_resnet50_fpn(num_classes=num_classes)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 6) Training loop
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        running_loss = 0.0

        for i, (images, targets) in enumerate(tqdm(loader, desc="Batches"), 1):
            # Move inputs to device
            images = [img.to(device) for img in images]
            moved_targets = []
            for t in targets:
                moved = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in t.items()
                }
                moved_targets.append(moved)

            # Forward + backward
            loss_dict = model(images, moved_targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            if i % 50 == 0:
                print(f"  Batch {i}/{len(loader)}  Loss: {losses.item():.4f}")

        avg = running_loss / len(loader)
        print(f"→ Epoch {epoch} done • Avg loss: {avg:.4f}")
        torch.save(model.state_dict(), f"maskrcnn_epoch{epoch}.pth")

    # Final checkpoint
    torch.save(model.state_dict(), "maskrcnn_final.pth")
    print("✅ Training complete — model saved as maskrcnn_final.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir",      default="objs",  help="Directory of .obj files")
    parser.add_argument("--epochs",     type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--device",     choices=["cpu","cuda","mps"], default=None)
    parser.add_argument("--smoke_test", action="store_true", help="Use small subset for testing")
    parser.add_argument("--views_per_obj", type=int, default=1,
                        help="Random views per mesh per epoch")
    args = parser.parse_args()

    # Device selection
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
        views_per_obj=args.views_per_obj
    )
