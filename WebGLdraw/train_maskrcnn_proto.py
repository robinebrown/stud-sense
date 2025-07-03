import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from synthetic_bricks_proto import SyntheticBrickProtoDataset, PART_IDS


def collate_fn(batch):
    return tuple(zip(*batch))

class MaskDataset(Dataset):
    """
    Wraps the prototype dataset to generate full Mask R-CNN targets.
    """
    def __init__(self, proto_ds):
        self.proto_ds = proto_ds
        self.num_parts = len(proto_ds.mesh_paths)

    def __len__(self):
        return len(self.proto_ds)

    def __getitem__(self, idx):
        image, meta = self.proto_ds[idx]
        mask = meta['mask']  # [H, W] uint8 mask

        # Compute bounding box
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel():
            y0, y1 = ys.min().item(), ys.max().item() + 1
            x0, x1 = xs.min().item(), xs.max().item() + 1
        else:
            y0, y1, x0, x1 = 0, 1, 0, 1
        boxes = torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32)

        # Label = part index + 1 (0 reserved for background)
        part_id = idx % self.num_parts
        labels = torch.tensor([part_id + 1], dtype=torch.int64)

        # Masks: [N, H, W]
        masks = mask.unsqueeze(0)

        # Other targets
        image_id = torch.tensor([idx])
        area = torch.tensor([(x1 - x0) * (y1 - y0)], dtype=torch.float32)
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }
        return image, target


def get_model(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = maskrcnn_resnet50_fpn(pretrained=True)
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes)
    return model


def train(args):
    # Prepare dataset
    proto_ds = SyntheticBrickProtoDataset(
        obj_dir=args.obj_dir,
        part_ids=PART_IDS,
        image_size=args.image_size,
        render_scale=args.render_scale,
        views_per_obj=args.views_per_obj,
        device=args.device
    )
    train_ds = MaskDataset(proto_ds)
    data_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn
    )

    device = torch.device(args.device)
    num_classes = len(PART_IDS) + 1  # +1 for background
    model = get_model(num_classes)
    model.to(device)

    # Optimizer and LR scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        lr_scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(data_loader):.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"maskrcnn_proto_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print("Saved checkpoint:", ckpt_path)

    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on LEGO prototype dataset")
    parser.add_argument('--obj-dir', type=str, default='objs', help='directory of .obj files')
    parser.add_argument('--image-size', type=int, default=256, help='output image size (square)')
    parser.add_argument('--render-scale', type=int, default=4, help='supersampling factor')
    parser.add_argument('--views-per-obj', type=int, default=3, help='random views per part')
    parser.add_argument('--device', type=str, default='cuda', help='torch device')
    parser.add_argument('--epochs', type=int, default=15, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--num-workers', type=int, default=2, help='dataloader workers')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--output-dir', type=str, default='checkpoints_proto', help='where to save models')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
