# train_maskrcnn.py

import torch.multiprocessing as mp
# Enforce spawn start method for worker processes (safer on Linux/macOS)
mp.set_start_method('spawn', force=True)

import argparse
import os
import torch
# Enable CuDNN benchmark for fixed-size inputs
torch.backends.cudnn.benchmark = True

from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from synthetic_bricks import SyntheticBrickDataset

class MultiViewDataset(Dataset):
    """
    Wraps a base dataset to repeat each element views_per_obj times.
    """
    def __init__(self, base_dataset, views_per_obj):
        self.base = base_dataset
        self.views = views_per_obj

    def __len__(self):
        return len(self.base) * self.views

    def __getitem__(self, idx):
        # modulo ensures each mesh appears views_per_obj times
        return self.base[idx % len(self.base)]

def collate_fn(batch):
    return tuple(zip(*batch))

def parse_args():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on synthetic LEGO bricks")
    parser.add_argument('--obj_dir',       type=str, required=True,
                        help='directory containing .obj files')
    parser.add_argument('--epochs',        type=int, default=15)
    parser.add_argument('--batch_size',    type=int, default=12)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--device',        type=str, default='cuda')
    parser.add_argument('--views_per_obj', type=int, default=8)
    parser.add_argument('--num_workers',   type=int, default=32)
    parser.add_argument('--image_size',    type=int, default=256)
    parser.add_argument('--smoke_test',    action='store_true',
                        help='quick smoke-test on 200 meshes, 2 epochs, 1 view')
    return parser.parse_args()

def train(args):
    device = torch.device(args.device)

    # Prepare base dataset
    if args.smoke_test:
        print("→ [Smoke-test] using first 200 meshes")
        base_ds = SyntheticBrickDataset(
            obj_dir=args.obj_dir,
            image_size=args.image_size,
            max_meshes=200,
            device=args.device
        )
        views, epochs = 1, 2
    else:
        base_ds = SyntheticBrickDataset(
            obj_dir=args.obj_dir,
            image_size=args.image_size,
            device=args.device
        )
        views, epochs = args.views_per_obj, args.epochs

    ds = MultiViewDataset(base_ds, views)
    print(f"Dataset size: {len(ds)} samples ({len(base_ds)} meshes × {views} views)")

    # DataLoader with pin_memory, persistent_workers, prefetch
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )

    # Build Mask R-CNN
    num_classes = len(base_ds) + 1  # class 0 = background
    model = maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(device_type='cuda')

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for i, (images, targets) in enumerate(loader, 1):
            # non-blocking transfers
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()}
                       for t in targets]

            with torch.cuda.amp.autocast(device_type='cuda'):
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if i % 50 == 0:
                avg = running_loss / 50
                print(f"Batch {i}/{len(loader)}  Loss: {avg:.4f}")
                running_loss = 0.0

        print(f"Epoch {epoch}/{epochs} completed.")
        torch.save(model.state_dict(), f"maskrcnn_epoch_{epoch}.pth")

    print("Training complete! Saving final model to maskrcnn_final.pth")
    torch.save(model.state_dict(), "maskrcnn_final.pth")

if __name__ == '__main__':
    args = parse_args()
    print(f"Training on device: {args.device}")
    train(args)
