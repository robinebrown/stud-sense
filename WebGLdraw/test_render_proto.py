#!/usr/bin/env python3
import os
import math
import random
import argparse
import torch
from torch.utils.data import Dataset
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    DirectionalLights,
    TexturesVertex
)
from torchvision.transforms import functional as F
import torchvision.utils as vutils


class SyntheticBrickDatasetV2(Dataset):
    """
    Standalone grayscale brick render dataset.
    Renders each LEGO part in constant light gray (0.85) under a headlight,
    with multiple random views, tight crops, and adjustable render resolution.
    """
    def __init__(
        self,
        obj_dir,
        image_size=330,
        render_scale=3,
        max_meshes=None,
        views_per_obj=1,
        device="cuda",
        camera_scale=2.5,
        fov=60.0
    ):
        self.device = torch.device(device)
        self.image_size = image_size
        self.render_size = image_size * render_scale
        self.views_per_obj = views_per_obj
        self.camera_scale = camera_scale
        self.fov = fov

        # Collect OBJ paths
        if os.path.isfile(obj_dir) and obj_dir.lower().endswith('.obj'):
            paths = [obj_dir]
        else:
            paths = [
                os.path.join(obj_dir, fn)
                for fn in os.listdir(obj_dir)
                if fn.lower().endswith('.obj')
            ]
        if max_meshes:
            paths = paths[:max_meshes]
        print(f"→ Found {len(paths)} .obj files in {obj_dir}")
        self.mesh_paths = paths

        # Load meshes & compute radii
        self.meshes = load_objs_as_meshes(self.mesh_paths, device=self.device)
        self.radii = []
        for verts in self.meshes.verts_list():
            center = verts.mean(0, keepdim=True)
            self.radii.append(((verts - center).norm(dim=1)).max().item())

        # Apply constant light gray textures
        gray = 0.85
        feats = [torch.full((v.shape[0], 3), gray, device=self.device)
                 for v in self.meshes.verts_list()]
        self.meshes.textures = TexturesVertex(verts_features=feats)

        # Renderer setup: headlight only
        self.cameras = FoVPerspectiveCameras(device=self.device, fov=self.fov)
        rast = RasterizationSettings(
            image_size=self.render_size,
            blur_radius=0.0,
            faces_per_pixel=1
        )
        default_lights = DirectionalLights(
            device=self.device,
            direction=[[0.0, 0.0, -1.0]],
            ambient_color=[[0.1, 0.1, 0.1]],
            diffuse_color=[[0.8, 0.8, 0.8]],
            specular_color=[[0.3, 0.3, 0.3]]
        )
        self.renderer = MeshRenderer(
            MeshRasterizer(cameras=self.cameras, raster_settings=rast),
            SoftPhongShader(device=self.device, cameras=self.cameras, lights=default_lights)
        )

    def __len__(self):
        return len(self.mesh_paths) * self.views_per_obj

    def __getitem__(self, idx):
        mesh_idx = idx % len(self.mesh_paths)
        mesh = self.meshes[mesh_idx]
        rad  = self.radii[mesh_idx]

        # Random camera pose
        half_fov = math.radians(self.fov / 2)
        dist = rad * self.camera_scale / math.tan(half_fov)
        azim = random.uniform(0, 360)
        elev = random.uniform(-75, 75)
        R, T = look_at_view_transform(dist, elev, azim, device=self.device)
        cam = FoVPerspectiveCameras(device=self.device, fov=self.fov, R=R, T=T)
        self.renderer.rasterizer.cameras = cam
        self.renderer.shader.cameras    = cam

        # Align headlight
        cam_to_world = R[0].T
        dir_cam = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=cam_to_world.dtype).view(1, 3)
        head_dir = (dir_cam @ cam_to_world).view(1, 3).tolist()
        self.renderer.shader.lights = DirectionalLights(
            device=self.device,
            direction=head_dir,
            ambient_color=[[0.1, 0.1, 0.1]],
            diffuse_color=[[0.8, 0.8, 0.8]],
            specular_color=[[0.3, 0.3, 0.3]]
        )

        # Render & mask
        out = self.renderer(mesh, R=R, T=T)
        rgb = out[0, ..., :3]
        frags = self.renderer.rasterizer(mesh, R=R, T=T)
        mask = (frags.pix_to_face[..., 0] >= 0).squeeze(0).to(torch.uint8)

        # Composite gray over black
        comp = rgb * mask.unsqueeze(-1).float()

        # Crop to bounds + margin
        ys, xs = torch.nonzero(mask, as_tuple=True)
        H, W = mask.shape
        if ys.numel():
            m = 16
            y0 = max(0, ys.min().item() - m)
            y1 = min(H, ys.max().item() + m)
            x0 = max(0, xs.min().item() - m)
            x1 = min(W, xs.max().item() + m)
        else:
            y0, y1, x0, x1 = 0, 1, 0, 1
        crop = comp[y0:y1, x0:x1, :].permute(2, 0, 1)
        mask_crop = mask[y0:y1, x0:x1]

        # Pad to square
        c, h, w = crop.shape
        if h > w:
            diff = h - w
            pad = (diff // 2, 0, diff - diff // 2, 0)
        else:
            diff = w - h
            pad = (0, diff // 2, 0, diff // 2)
        crop = F.pad(crop, pad, fill=0)
        mask_crop = F.pad(mask_crop.unsqueeze(0), pad, fill=0).squeeze(0)

        # Downsample to final size
        image = F.resize(crop, [self.image_size, self.image_size])
        mask_out = F.resize(
            mask_crop.unsqueeze(0).float(),
            [self.image_size, self.image_size],
            interpolation=F.InterpolationMode.NEAREST
        )[0].to(torch.uint8)

        return image, {'mask': mask_out}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Render a single LEGO .obj into multiple prototype views."
    )
    parser.add_argument(
        "--obj_path", type=str, required=True,
        help="Path to the single .obj file (e.g. objs/3001.obj)"
    )
    parser.add_argument(
        "--views", type=int, default=5,
        help="How many random views to render per part (default=5)"
    )
    parser.add_argument(
        "--size", type=int, default=330,
        help="Final output image size (square, default=330)"
    )
    parser.add_argument(
        "--render_scale", type=int, default=3,
        help="Supersampling factor (default=3)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device: cpu, mps, or cuda"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Directory to save renders and masks"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Instantiate dataset for just this OBJ
    ds = SyntheticBrickDatasetV2(
        obj_dir=args.obj_path,
        image_size=args.size,
        render_scale=args.render_scale,
        views_per_obj=args.views,
        device=args.device
    )

    # Derive part ID for naming
    pid = os.path.splitext(os.path.basename(args.obj_path))[0]

    # Render and save
    for idx in range(len(ds)):
        image, meta = ds[idx]
        view_num = idx + 1
        img_name = f"{pid}_{view_num:02d}.png"
        mask_name = f"{pid}_{view_num:02d}_mask.png"
        vutils.save_image(image, os.path.join(args.out_dir, img_name))
        vutils.save_image(meta['mask'].unsqueeze(0).float(),
                          os.path.join(args.out_dir, mask_name))

    print(f"Rendered {len(ds)} views for {pid} into {args.out_dir}")
