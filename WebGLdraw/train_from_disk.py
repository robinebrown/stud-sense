# train_from_disk.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from torch.optim.lr_scheduler import MultiStepLR

class PrerenderedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.img_dir = os.path.join(root_dir, "images")
        self.ann_dir = os.path.join(root_dir, "annotations")
        # list all base names
        self.ids = sorted([fname[:-4]
                           for fname in os.listdir(self.img_dir)
                           if fname.endswith(".png") and "_mask" not in fname])
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        base = self.ids[idx]
        # load RGB
        img = Image.open(f"{self.img_dir}/{base}.png").convert("RGB")
        image = F.to_tensor(img)  # [3,H,W], float 0–1
        # load mask
        mask = Image.open(f"{self.img_dir}/{base}_mask.png")
        mask = torch.as_tensor(np.array(mask), dtype=torch.uint8)
        # load boxes/labels
        ann = torch.load(f"{self.ann_dir}/{base}.pt")
        boxes  = ann["boxes"]
        labels = ann["labels"]

        target = {
            "boxes":  boxes,
            "labels": labels,
            "masks":  mask.unsqueeze(0),
            "image_id": torch.tensor([idx]),
            "area":   (boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return image, target

def collate_fn(batch):
    imgs, tars = zip(*batch)
    return list(imgs), list(tars)

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--epochs",     type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--num_workers",type=int, default=8)
    return p.parse_args()

def train(args):
    ds = PrerenderedDataset(args.data_dir)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    device = torch.device(args.device)
    model = maskrcnn_resnet50_fpn(num_classes=len(ds)+1).to(device).train()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler= GradScaler()
    sched = MultiStepLR(optim, milestones=[8,12], gamma=0.1)

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        running = 0.0
        for i,(imgs, tars) in enumerate(tqdm(loader),1):
            imgs_gpu = [im.to(device, non_blocking=True) for im in imgs]
            tars_gpu = [{k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                         for k,v in tar.items()} for tar in tars]

            with autocast():
                loss_dict = model(imgs_gpu, tars_gpu)
                loss = sum(loss_dict.values())

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running += loss.item()
            if i%50==0:
                print(f"  Batch {i}/{len(loader)} Loss: {running/50:.4f}")
                running=0.0

        sched.step()
        torch.save(model.state_dict(), f"maskrcnn_epoch{epoch}.pth")

    torch.save(model.state_dict(), "maskrcnn_final.pth")
    print("Training complete! Model saved to maskrcnn_final.pth")

if __name__=="__main__":
    args = parse_args()
    train(args)
