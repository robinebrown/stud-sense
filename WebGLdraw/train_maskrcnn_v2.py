import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch, argparse
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from synthetic_bricks_v2 import SyntheticBrickDatasetV2
from tqdm import tqdm
from torch.amp import autocast, GradScaler

if __name__ == "__main__":
    cudnn.benchmark = True
    p = argparse.ArgumentParser()
    p.add_argument("--obj_dir",      default="objs")
    p.add_argument("--epochs",       type=int, default=30)
    p.add_argument("--batch_size",   type=int, default=12)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--device",       default=None)
    p.add_argument("--smoke_test",   action="store_true")
    p.add_argument("--views_per_obj",type=int, default=8)
    p.add_argument("--image_size",   type=int, default=256)
    p.add_argument("--num_workers",  type=int, default=8)
    p.add_argument("--colors_csv",   default="colors.csv")
    p.add_argument("--elements_csv", default="elements.csv")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)

    ds = SyntheticBrickDatasetV2(
        obj_dir=args.obj_dir,
        image_size=args.image_size,
        max_meshes=200 if args.smoke_test else None,
        device=device,
        views_per_obj=args.views_per_obj,
        colors_csv=args.colors_csv,
        elements_csv=args.elements_csv
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: list(zip(*b)),
                        num_workers=args.num_workers, pin_memory=True,
                        prefetch_factor=(4 if args.num_workers>0 else None),
                        persistent_workers=(args.num_workers>0))

    model = maskrcnn_resnet50_fpn(num_classes=len(ds)+1).to(device).train()
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler= GradScaler()

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        loss_accum = 0.0
        for i,(imgs, tars) in enumerate(tqdm(loader),1):
            imgs   = [img.to(device) for img in imgs]
            targets= [{k:(v.to(device) if torch.is_tensor(v) else v) for k,v in tar.items()} for tar in tars]

            with autocast():
                losses = model(imgs, targets)
                loss = sum(losses.values())

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_accum += loss.item()
            if i%50==0:
                print(f"  Batch {i}/{len(loader)}  Loss {loss_accum/50:.4f}")
                loss_accum = 0.0

    torch.save(model.state_dict(), "maskrcnn_v2_final.pth")
    print("Done.")
