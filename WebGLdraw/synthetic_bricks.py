# synthetic_bricks.py

import os
import random
import torch
from torch.utils.data import Dataset
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointLights,
    TexturesVertex
)
import kornia.augmentation as K

def random_spherical_pose(device):
    theta = torch.rand(1, device=device) * 2 * torch.pi
    phi   = torch.acos(2 * torch.rand(1, device=device) - 1)
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    R = torch.eye(3, device=device).unsqueeze(0)
    T = torch.stack([x, y, z], dim=1)
    return R, T

class SyntheticBrickDataset(Dataset):
    def __init__(self, obj_dir, image_size=256, max_tries=5, max_meshes=None, device="cuda"):
        self.device = torch.device(device)

        # 1) scan .obj files
        all_paths = [
            os.path.join(obj_dir, f)
            for f in os.listdir(obj_dir)
            if f.lower().endswith('.obj')
        ]
        if max_meshes is not None:
            all_paths = all_paths[:max_meshes]
        print(f"→ Found {len(all_paths)} .obj files in {obj_dir}")

        # 2) load meshes once onto GPU
        meshes = load_objs_as_meshes(all_paths, device=self.device)
        verts_list = meshes.verts_list()
        valid_idx = [i for i, v in enumerate(verts_list) if v.shape[0] > 0]
        if len(valid_idx) < len(verts_list):
            print(f"   · filtered out {len(verts_list) - len(valid_idx)} empty meshes")
        self.meshes = meshes[valid_idx]

        # 3) assign white vertex textures
        verts_list = self.meshes.verts_list()
        feats = [torch.ones((v.shape[0], 3), device=self.device) for v in verts_list]
        self.meshes.textures = TexturesVertex(verts_features=feats)

        # 4) build GPU-side renderer
        cams = FoVPerspectiveCameras(device=self.device)
        rast_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
            max_faces_per_bin=1000
        )
        lights = PointLights(device=self.device, location=[[0, 0, 3]])
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cams, raster_settings=rast_settings),
            shader=SoftPhongShader(device=self.device, cameras=cams, lights=lights)
        )
        # force renderer onto GPU
        self.renderer = self.renderer.to(self.device)

        # 5) GPU augmentations (applied to both image & mask where appropriate)
        self.augment = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomRotation(degrees=15.0, resample='BILINEAR'),
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            data_keys=["input", "mask"]
        )

        self.max_tries = max_tries

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):
        mesh = self.meshes[idx]

        # render until visible
        for _ in range(self.max_tries):
            R, T = random_spherical_pose(self.device)
            rgb = self.renderer(mesh, R=R, T=T)[0, ..., :3]
            frags = self.renderer.rasterizer(mesh, R=R, T=T)
            mask = frags.pix_to_face[..., 0] >= 0
            if mask.any():
                break
        else:
            # fallback centered view
            R = torch.eye(3, device=self.device).unsqueeze(0)
            T = torch.tensor([[0, 0, 3]], device=self.device, dtype=torch.float32)
            rgb = self.renderer(mesh, R=R, T=T)[0, ..., :3]
            frags = self.renderer.rasterizer(mesh, R=R, T=T)
            mask = frags.pix_to_face[..., 0] >= 0

        # prepare tensors with batch dim for Kornia
        image = rgb.permute(2, 0, 1).unsqueeze(0)       # (1,3,H,W)
        mask  = mask.to(torch.float32).unsqueeze(1)     # (1,1,H,W)

        # apply GPU augmentations
        image_aug, mask_aug = self.augment(image, mask)

        # threshold mask back to binary and remove batch/channel dims
        mask_aug = (mask_aug > 0.5).to(torch.uint8).squeeze(0).squeeze(0)  # [H,W]
        image = image_aug.squeeze(0)                                     # [3,H,W]

        # compute bounding box from mask
        ys, xs = torch.nonzero(mask_aug, as_tuple=True)
        if ys.numel() == 0:
            ymin, ymax, xmin, xmax = 0, 1, 0, 1
        else:
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()
            xmax = max(xmax, xmin + 1)
            ymax = max(ymax, ymin + 1)

        box = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32, device=self.device)
        target = {
            "boxes":    box,
            "labels":   torch.tensor([idx + 1], dtype=torch.int64, device=self.device),
            "masks":    mask_aug.unsqueeze(0),
            "image_id": torch.tensor([idx], device=self.device),
            "area":     (xmax - xmin) * (ymax - ymin),
            "iscrowd":  torch.zeros(1, dtype=torch.int64, device=self.device)
        }

        return image, target
