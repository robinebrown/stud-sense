import os
import glob
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from synthetic_bricks import SyntheticBrickDataset
from tqdm import tqdm

# ─── Ensure CUDA works in subprocesses ───────────────────────────────────────────
if torch.cuda.is_available():
    mp.set_start_method('spawn', force=True)

class MultiViewDataset:
    """Repeats each object multiple times for multiple random views."""
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
    all_objs = sorted(glob.glob(os.path.join(obj_dir, "*.obj")))
    print(f"→ Found {len(all_objs)} .obj files in {obj_dir}")
    if smoke_test:
        all_objs = all_objs[:200]
        print(f"→ [Smoke-test] Using {len(all_objs)} files")

    # 2) Build dataset
    base_ds = SyntheticBrickDataset(
        obj_dir=obj_dir,
        image_size=256,
        max_tries=5,
        smoke_test=smoke_test
    )

    # 3) Wrap for multi-view
    if views_per_obj > 1:
        ds = MultiViewDataset(base_ds, views_per_obj)
        print(f"→ Multi-view: {views_per_obj}× → dataset size {len(ds)}")
    else:
        ds = base_ds

    # 4) DataLoader (single-process for CUDA safety)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,        # ← single‐process ensures no CUDA‐fork conflicts
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 5) Model & optimizer
    num_classes = len(base_ds) + 1
    model = maskrcnn_resnet50_fpn(num_classes=num_classes)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 6) Training loop
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")
        epoch_loss = 0.0

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

            epoch_loss += losses.item()
            if i % 50 == 0:
                print(f"  Batch {i}/{len(loader)}  Loss: {losses.item():.4f}")

        avg = epoch_loss / len(loader)
        print(f"→ Epoch {epoch} done • Avg loss: {avg:.4f}")
        torch.save(model.state_dict(), f"maskrcnn_epoch{epoch}.pth")

    # Final checkpoint
    torch.save(model.state_dict(), "maskrcnn_final.pth")
    print("✅ Smoke-test training complete — model saved as maskrcnn_final.pth")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--obj_dir",      default="objs",   help="Folder with .obj meshes")
    p.add_argument("--epochs",     type=int, default=1,  help="Number of epochs")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--device",     choices=["cpu","cuda","mps"], default=None)
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--views_per_obj", type=int, default=1)
    args = p.parse_args()

    # Device selection
    if args.device:
        dev = torch.device(args.device)
    else:
        if   torch.cuda.is_available(): dev = torch.device("cuda")
        elif torch.backends.mps.is_available(): dev = torch.device("mps")
        else:                              dev = torch.device("cpu")
    print("Training on device:", dev)

    train(
      obj_dir=args.obj_dir,
      epochs=args.epochs,
      batch_size=args.batch_size,
      lr=args.lr,
      device=dev,
      smoke_test=args.smoke_test,
      views_per_obj=args.views_per_obj
    )
