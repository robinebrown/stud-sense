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

class SyntheticBrickDataset(Dataset):
    """
    Grayscale renders with headlight illumination for consistent 3D shading:
      - White piece, black background
      - Directional light aligned with camera view (headlight)
      - Random camera azimuth [0–360°], elevation [15–75°]
      - Crop to object, pad to square, resize without distortion
      - Mild contrast boost + slight 2D rotations/flips
    """
    def __init__(
        self,
        obj_dir,
        image_size=256,
        max_tries=5,
        max_meshes=None,
        device="cuda",
        camera_scale=2.5,
        fov=60.0
    ):
        self.device = torch.device(device)
        self.camera_scale = camera_scale
        self.fov = fov
        self.image_size = image_size
        self.render_size = image_size * 2
        self.max_tries = max_tries

        # 1) Gather .obj paths
        if os.path.isfile(obj_dir) and obj_dir.lower().endswith('.obj'):
            paths = [obj_dir]
        else:
            paths = [
                os.path.join(obj_dir, f)
                for f in os.listdir(obj_dir)
                if f.lower().endswith('.obj')
            ]
        if max_meshes:
            paths = paths[:max_meshes]
        print(f"→ Found {len(paths)} .obj files in {obj_dir}")

        # 2) Load meshes
        meshes = load_objs_as_meshes(paths, device=self.device)
        valid = [i for i, v in enumerate(meshes.verts_list()) if v.shape[0] > 0]
        self.meshes = meshes[valid]

        # 3) Compute bounding radii
        self.radii = []
        for verts in self.meshes.verts_list():
            center = verts.mean(0, keepdim=True)
            radius = (verts - center).norm(dim=1).max().item()
            self.radii.append(radius)

        # 4) Assign white textures
        feats = [torch.ones((v.shape[0], 3), device=self.device)
                 for v in self.meshes.verts_list()]
        self.meshes.textures = TexturesVertex(verts_features=feats)

        # 5) Renderer placeholder
        base_cam = FoVPerspectiveCameras(device=self.device, fov=self.fov)
        rast_settings = RasterizationSettings(
            image_size=self.render_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
            max_faces_per_bin=1000
        )
        default_lights = DirectionalLights(
            device=self.device,
            direction=[[0.0, 0.0, -1.0]],  # will override per sample
            ambient_color=[[0.05, 0.05, 0.05]],
            diffuse_color=[[1.0, 1.0, 1.0]],
            specular_color=[[0.5, 0.5, 0.5]]
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=base_cam, raster_settings=rast_settings),
            shader=SoftPhongShader(device=self.device, cameras=base_cam, lights=default_lights)
        ).to(self.device)

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):
        mesh = self.meshes[idx]
        radius = self.radii[idx]

        # 1) Random camera pose
        azim = random.uniform(0, 360)
        elev = random.uniform(-75, 75)  # allow views from underside, sides, and top
        half_fov = math.radians(self.fov / 2)
        dist = (radius * self.camera_scale) / math.tan(half_fov)
        R, T = look_at_view_transform(dist, elev, azim, device=self.device)
        cam = FoVPerspectiveCameras(device=self.device, fov=self.fov, R=R, T=T)
        self.renderer.rasterizer.cameras = cam
        self.renderer.shader.cameras    = cam

        # 2) Headlight: light direction aligned with camera view
        # Compute camera-to-world rotation
        cam_to_world = R[0].T  # (3,3)
        # In camera coords, headlight comes along -Z
        dir_cam = torch.tensor([0.0, 0.0, -1.0], device=self.device).view(1,3)
        # Transform to world coords
        dir_world = (dir_cam @ cam_to_world).cpu().tolist()
        lights = DirectionalLights(
            device=self.device,
            direction=dir_world,
            ambient_color=[[0.1, 0.1, 0.1]],
            diffuse_color=[[0.8, 0.8, 0.8]],
            specular_color=[[0.3, 0.3, 0.3]]
        )
        self.renderer.shader.lights = lights

        # 3) Render
        out = self.renderer(mesh, R=R, T=T)
        rgb  = out[0, ..., :3]
        frags = self.renderer.rasterizer(mesh, R=R, T=T)
        mask = (frags.pix_to_face[..., 0] >= 0).squeeze(0)

        # 4) Convert to tensors
        image = rgb.permute(2, 0, 1)      # (3, H, W)
        mask_t = mask.to(torch.uint8)    # (H, W)

        # 5) Crop & pad to square
        H, W = mask_t.shape
        ys, xs = torch.nonzero(mask_t, as_tuple=True)
        if ys.numel():
            m = 16
            y0 = max(0, ys.min().item() - m)
            y1 = min(H, ys.max().item() + m)
            x0 = max(0, xs.min().item() - m)
            x1 = min(W, xs.max().item() + m)
        else:
            y0, y1, x0, x1 = 0, 1, 0, 1
        crop_img = F.crop(image, y0, x0, y1 - y0, x1 - x0)
        crop_mask = F.crop(mask_t.unsqueeze(0), y0, x0, y1 - y0, x1 - x0).squeeze(0)
        ch, cw = crop_img.shape[1:]
        if ch > cw:
            diff = ch - cw
            pad = (diff//2, 0, diff - diff//2, 0)
        else:
            diff = cw - ch
            pad = (0, diff//2, 0, diff - diff//2)
        crop_img = F.pad(crop_img, pad, fill=0)
        crop_mask = F.pad(crop_mask.unsqueeze(0), pad, fill=0).squeeze(0)
        image = F.resize(crop_img, [self.image_size, self.image_size], interpolation=F.InterpolationMode.BILINEAR)
        mask  = F.resize(crop_mask.unsqueeze(0).float(), [self.image_size, self.image_size], interpolation=F.InterpolationMode.NEAREST)[0].to(torch.uint8)

        # 6) Black background
        image = torch.where(mask.unsqueeze(0) == 1, image, torch.zeros_like(image))

        # 7) Mild contrast boost
        cvar = random.uniform(1.0, 1.5)
        image = F.adjust_contrast(image, cvar).clamp(0, 1)

        # 8) 2D rotation & flips
        rot = random.uniform(-10, 10)
        image = F.rotate(image, rot, interpolation=F.InterpolationMode.BILINEAR, fill=(0,0,0))
        mask  = F.rotate(mask.unsqueeze(0), rot, interpolation=F.InterpolationMode.NEAREST, fill=0)[0].to(torch.uint8)
        if random.random() < 0.5:
            image = F.hflip(image); mask = F.hflip(mask)
        if random.random() < 0.5:
            image = F.vflip(image); mask = F.vflip(mask)

        # 9) Bounding box
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel():
            y0, y1 = ys.min().item(), ys.max().item() + 1
            x0, x1 = xs.min().item(), xs.max().item() + 1
        else:
            y0, y1, x0, x1 = 0, 1, 0, 1
        box = torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32, device=self.device)
        target = {
            'boxes':    box,
            'labels':   torch.tensor([idx+1], dtype=torch.int64, device=self.device),
            'masks':    mask.unsqueeze(0),
            'image_id': torch.tensor([idx], device=self.device),
            'area':     (x1 - x0) * (y1 - y0),
            'iscrowd':  torch.zeros(1, dtype=torch.int64, device=self.device)
        }
        return image, target
