import os
import math
import random
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

# Default list of 20 prototype part IDs
PART_IDS = [
    "3001","3003","3020","3022","3040","54200","3665","3062","4740","15254",
    "4865","32000","2780","3626","2555","6238","51239","88293","2817","30414"
]

class SyntheticBrickProtoDataset(Dataset):
    """
    Dataset for rendering a fixed set of prototype parts in light gray.
    """
    def __init__(self,
                 obj_dir,
                 part_ids,
                 image_size=330,
                 render_scale=3,
                 views_per_obj=1,
                 device="cpu",
                 camera_scale=2.5,
                 fov=60.0):
        self.device = torch.device(device)
        self.image_size = image_size
        self.render_size = image_size * render_scale
        self.views_per_obj = views_per_obj
        self.camera_scale = camera_scale
        self.fov = fov

        # Build full paths to each part's .obj
        self.mesh_paths = [os.path.join(obj_dir, f"{pid}.obj") for pid in part_ids]
        print(f"→ Prototype parts: {len(self.mesh_paths)} meshes from {obj_dir}")

        # Load meshes
        self.meshes = load_objs_as_meshes(self.mesh_paths, device=self.device)
        self.radii = []
        for verts in self.meshes.verts_list():
            center = verts.mean(0, keepdim=True)
            self.radii.append(((verts - center).norm(dim=1)).max().item())

        # Assign constant light gray vertex color
        gray = 0.8
        feats = [torch.full((v.shape[0], 3), gray, device=self.device)
                 for v in self.meshes.verts_list()]
        self.meshes.textures = TexturesVertex(verts_features=feats)

        # Renderer: headlight only
        self.cameras = FoVPerspectiveCameras(device=self.device, fov=self.fov)
        rast = RasterizationSettings(
            image_size=self.render_size,
            blur_radius=0.0,
            faces_per_pixel=1
        )
        lights = DirectionalLights(
            device=self.device,
            direction=[[0.0, 0.0, -1.0]],
            ambient_color=[[0.1, 0.1, 0.1]],
            diffuse_color=[[0.8, 0.8, 0.8]],
            specular_color=[[0.3, 0.3, 0.3]]
        )
        self.renderer = MeshRenderer(
            MeshRasterizer(cameras=self.cameras, raster_settings=rast),
            SoftPhongShader(device=self.device, cameras=self.cameras, lights=lights)
        )

    def __len__(self):
        return len(self.mesh_paths) * self.views_per_obj

    def __getitem__(self, idx):
        mesh_idx = idx % len(self.mesh_paths)
        mesh = self.meshes[mesh_idx]
        rad = self.radii[mesh_idx]

        # Random camera pose
        half = math.radians(self.fov / 2)
        dist = rad * self.camera_scale / math.tan(half)
        azim = random.uniform(0, 360)
        elev = random.uniform(-75, 75)
        R, T = look_at_view_transform(dist, elev, azim, device=self.device)
        cam = FoVPerspectiveCameras(device=self.device, fov=self.fov, R=R, T=T)
        self.renderer.rasterizer.cameras = cam
        self.renderer.shader.cameras = cam

        # Align headlight
        cam_to_world = R[0].T
        dir_cam = torch.tensor([0.0, 0.0, -1.0], device=self.device,
                               dtype=cam_to_world.dtype).view(1, 3)
        world_dir = (dir_cam @ cam_to_world).view(1, 3).tolist()
        self.renderer.shader.lights = DirectionalLights(
            device=self.device,
            direction=world_dir,
            ambient_color=[[0.1, 0.1, 0.1]],
            diffuse_color=[[0.8, 0.8, 0.8]],
            specular_color=[[0.3, 0.3, 0.3]]
        )

        # Render and get mask
        out = self.renderer(mesh, R=R, T=T)
        img = out[0, ..., :3]  # [H, W, 3]
        frags = self.renderer.rasterizer(mesh, R=R, T=T)
        mask = (frags.pix_to_face[..., 0] >= 0).squeeze(0).to(torch.uint8)

        # Composite gray over black
        comp = img * mask.unsqueeze(-1).float()

        # Crop to mask bounds + margin
        ys, xs = torch.nonzero(mask, as_tuple=True)
        H, W = mask.shape
        if ys.numel():
            m = 16
            y0, y1 = max(0, ys.min().item() - m), min(H, ys.max().item() + m)
            x0, x1 = max(0, xs.min().item() - m), min(W, xs.max().item() + m)
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
            pad = (0, diff // 2, 0, diff - diff // 2)
        crop = F.pad(crop, pad, fill=0)
        mask_crop = F.pad(mask_crop.unsqueeze(0), pad, fill=0).squeeze(0)

        # === Add extra 20px border padding ===
        border = 45
        crop = F.pad(crop, (border, border, border, border), fill=0)
        mask_crop = F.pad(mask_crop.unsqueeze(0), (border, border, border, border), fill=0).squeeze(0)

        # Downsample
        image = F.resize(crop, [self.image_size, self.image_size])
        mask_out = F.resize(
            mask_crop.unsqueeze(0).float(),
            [self.image_size, self.image_size],
            interpolation=F.InterpolationMode.NEAREST
        )[0].to(torch.uint8)

        return image, {'mask': mask_out}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Render prototype LEGO parts to images and masks.")
    parser.add_argument('--part-ids', type=str,
                        help='Comma-separated list of LDraw IDs to render (e.g. "3001,3020,3626").')
    parser.add_argument('--views_per_obj', type=int, default=1,
                        help='How many random views to render per part (default=1).')
    parser.add_argument('--size', type=int, default=330,
                        help='Output image size in pixels (square, default=330).')
    parser.add_argument('--render_scale', type=int, default=3,
                        help='Supersampling scale factor for rendering (default=3).')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device to use (cpu or cuda, default=cpu).')
    parser.add_argument('--out_dir', type=str, default='viz_outputs/proto',
                        help='Directory to save rendered images and masks.')
    args = parser.parse_args()

    # Determine which parts to render
    selected_ids = args.part_ids.split(',') if args.part_ids else PART_IDS

    # Initialize dataset
    ds = SyntheticBrickProtoDataset(
        obj_dir='objs',
        part_ids=selected_ids,
        image_size=args.size,
        render_scale=args.render_scale,
        views_per_obj=args.views_per_obj,
        device=args.device
    )
    os.makedirs(args.out_dir, exist_ok=True)

    # Number of parts for naming
    num_parts = len(selected_ids)

    # Render and save with partID_viewNumber naming
    for idx in range(len(ds)):
        img, meta = ds[idx]
        mesh_idx = idx % num_parts
        view_idx = idx // num_parts + 1
        pid = selected_ids[mesh_idx]
        img_name = f"{pid}_{view_idx:02d}.png"
        mask_name = f"{pid}_{view_idx:02d}_mask.png"
        vutils.save_image(img, os.path.join(args.out_dir, img_name))
        vutils.save_image(meta['mask'].unsqueeze(0).float(),
                          os.path.join(args.out_dir, mask_name))

    print(f"Rendered {len(ds)} samples to {args.out_dir}")
