import os, glob, torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from synthetic_bricks import SyntheticBrickDataset
from tqdm import tqdm

# ─── force spawn when using cuda ────────────────────────────────────────────────
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
    # gather all .obj
    all_objs = sorted(glob.glob(os.path.join(obj_dir, "*.obj")))
    print(f"→ Found {len(all_objs)} .obj files")
    if smoke_test:
        all_objs = all_objs[:200]
        print(f"→ [Smoke-test] Using {len(all_objs)} files")

    # base dataset
    base_ds = SyntheticBrickDataset(
        obj_dir=obj_dir,
        image_size=256,
        smoke_test=smoke_test
    )

    # wrap for multi-view
    if views_per_obj > 1:
        ds = MultiViewDataset(base_ds, views_per_obj)
        print(f"→ Multi-view: {views_per_obj}× → dataset size {len(ds)}")
    else:
        ds = base_ds

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # model & optimizer
    num_classes = len(base_ds) + 1
    model = maskrcnn_resnet50_fpn(num_classes=num_classes)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")
        epoch_loss = 0.0
        for i, (images, targets) in enumerate(tqdm(loader, desc="Batches"), 1):
            images = [img.to(device) for img in images]
            moved_targets = []
            for t in targets:
                moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k,v in t.items()}
                moved_targets.append(moved)

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

    torch.save(model.state_dict(), "maskrcnn_final.pth")
    print("✅ Training complete — model saved as maskrcnn_final.pth")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--obj_dir",      default="objs")
    p.add_argument("--epochs",     type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--device",     choices=["cpu","cuda","mps"])
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--views_per_obj", type=int, default=1)
    args = p.parse_args()

    # auto-select device
    if args.device:
        dev = torch.device(args.device)
    else:
        if   torch.cuda.is_available(): dev = torch.device("cuda")
        elif torch.backends.mps.is_available(): dev = torch.device("mps")
        else: dev = torch.device("cpu")
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
