# train_maskrcnn.py

import torch.multiprocessing as mp
# spawn + file_system sharing avoids FD exhaustion & CUDA-fork bugs
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

# use a fast 'fork' context inside DataLoader
fork_ctx = get_context('fork')

def collate_fn(batch):
    imgs, tars = zip(*batch)
    return list(imgs), list(tars)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--obj_dir",       default="objs")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=10)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--device",        default=None)
    p.add_argument("--smoke_test",    action="store_true")
    p.add_argument("--views_per_obj", type=int, default=8)
    p.add_argument("--image_size",    type=int, default=256)
    p.add_argument("--num_workers",   type=int, default=16)
    return p.parse_args()

def train(args):
    # 1) GPU-based rendering, CPU dataset
    ds = SyntheticBrickDataset(
        obj_dir=args.obj_dir,
        image_size=args.image_size,
        max_meshes=200 if args.smoke_test else None,
        device=args.device  # render on GPU
    )
    if args.smoke_test:
        print(f"→ [Smoke-test] using first {len(ds)} meshes")

    # 2) Repeat for multi-view
    if args.views_per_obj > 1:
        from torch.utils.data import Dataset
        class MultiView(Dataset):
            def __init__(self, base, v): self.base, self.v = base, v
            def __len__(self): return len(self.base)*self.v
            def __getitem__(self,i): return self.base[i % len(self.base)]
        dataset = MultiView(ds, args.views_per_obj)
        print(f"→ Multi-view: {args.views_per_obj}×, size={len(dataset)}")
    else:
        dataset = ds

    # 3) Fast DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        multiprocessing_context=fork_ctx,
        pin_memory=True,          # dataset yields CPU tensors
        prefetch_factor=2,        # queue 2 batches ahead
        persistent_workers=True,
        collate_fn=collate_fn
    )

    # 4) Model + AMP + scheduler
    num_classes = len(ds)+1
    model = maskrcnn_resnet50_fpn(num_classes=num_classes).to(args.device).train()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    sched  = MultiStepLR(optim, milestones=[8,12], gamma=0.1)

    # 5) Train loop
    for e in range(1, args.epochs+1):
        print(f"\nEpoch {e}/{args.epochs}")
        running = 0.0
        for i, (imgs, tars) in enumerate(tqdm(loader, desc="Batches"),1):
            # move batch to GPU
            imgs_gpu = [im.to(args.device, non_blocking=True) for im in imgs]
            tars_gpu = []
            for tar in tars:
                tg = {k:(v.to(args.device, non_blocking=True) if torch.is_tensor(v) else v)
                      for k,v in tar.items()}
                tars_gpu.append(tg)

            if e==1 and i==1:
                print("image[0].shape =", imgs_gpu[0].shape)

            with autocast():
                lossdict = model(imgs_gpu, tars_gpu)
                loss = sum(lossdict.values())

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running += loss.item()
            if i%50==0:
                print(f"  Batch {i}/{len(loader)}  Loss={running/50:.4f}")
                running = 0.0

        sched.step()
        torch.save(model.state_dict(), f"maskrcnn_epoch{e}.pth")

    torch.save(model.state_dict(), "maskrcnn_final.pth")
    print("Training complete — maskrcnn_final.pth saved")


if __name__=="__main__":
    args = parse_args()
    # pick device
    if args.device:
        dev = torch.device(args.device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = dev
    print("Training on:", dev)
    train(args)
