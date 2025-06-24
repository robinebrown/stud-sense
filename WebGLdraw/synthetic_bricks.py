import os
import math
import random
import colorsys
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
    Grayscale renders with optional color variation for robustness:
      - White or randomly colored piece on a black background
      - Headlight directional illumination aligned with camera
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
            self.radii.append((verts - center).norm(dim=1).max().item())

        # 4) Assign initial white textures
        feats = [torch.ones((v.shape[0], 3), device=self.device)
                 for v in self.meshes.verts_list()]
        self.meshes.textures = TexturesVertex(verts_features=feats)

        # 5) Renderer placeholder setup
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
            direction=[[0.0, 0.0, -1.0]],  # headlight direction override per sample
            ambient_color=[[0.1, 0.1, 0.1]],
            diffuse_color=[[0.8, 0.8, 0.8]],
            specular_color=[[0.3, 0.3, 0.3]]
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=base_cam, raster_settings=rast_settings),
            shader=SoftPhongShader(device=self.device, cameras=base_cam, lights=default_lights)
        ).to(self.device)

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):
        # 1) Select the mesh and optionally apply color variation
        mesh = self.meshes[idx]
        # 45% chance to tint the piece randomly
        if random.random() < 0.45:
            h = random.random()
            s = random.uniform(0.5, 1.0)
            v = random.uniform(0.6, 1.0)
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            verts = mesh.verts_list()[0]
            color_feat = torch.tensor([r, g, b], device=self.device, dtype=torch.float32)
            feats = color_feat.unsqueeze(0).repeat(verts.shape[0], 1)
            mesh.textures = TexturesVertex(verts_features=[feats])

        # 2) Sample random camera pose
        radius = self.radii[idx]
        azim = random.uniform(0, 360)
        elev = random.uniform(-75, 75)  # full range: underside, sides, top
        half_fov = math.radians(self.fov / 2)
        dist = (radius * self.camera_scale) / math.tan(half_fov)
        R, T = look_at_view_transform(dist, elev, azim, device=self.device)
        cam = FoVPerspectiveCameras(device=self.device, fov=self.fov, R=R, T=T)
        self.renderer.rasterizer.cameras = cam
        self.renderer.shader.cameras    = cam

        # 3) Headlight: align directional light with camera forward vector
        cam_to_world = R[0].T  # invert rotation
        dir_cam = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).view(1, 3)
        dir_world = (dir_cam @ cam_to_world).cpu().tolist()
        lights = DirectionalLights(
            device=self.device,
            direction=dir_world,
            ambient_color=[[0.1, 0.1, 0.1]],
            diffuse_color=[[0.8, 0.8, 0.8]],
            specular_color=[[0.3, 0.3, 0.3]]
        )
        self.renderer.shader.lights = lights

        # 4) Render RGB + mask
        out = self.renderer(mesh, R=R, T=T)
        rgb = out[0, ..., :3]
        frags = self.renderer.rasterizer(mesh, R=R, T=T)
        mask = (frags.pix_to_face[..., 0] >= 0).squeeze(0)

        # 5) Convert to tensors
        image = rgb.permute(2, 0, 1)    # (3, H, W)
        mask = mask.to(torch.uint8)     # (H, W)

        # 6) Crop & pad to square
        H, W = mask.shape
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel():
            m = 16
            y0 = max(0, ys.min().item() - m)
            y1 = min(H, ys.max().item() + m)
            x0 = max(0, xs.min().item() - m)
            x1 = min(W, xs.max().item() + m)
        else:
            y0, y1, x0, x1 = 0, 1, 0, 1
        crop_img = F.crop(image, y0, x0, y1 - y0, x1 - x0)
        crop_mask = F.crop(mask.unsqueeze(0), y0, x0, y1 - y0, x1 - x0).squeeze(0)
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
        mask = F.resize(crop_mask.unsqueeze(0).float(), [self.image_size, self.image_size], interpolation=F.InterpolationMode.NEAREST)[0].to(torch.uint8)

        # 7) Black background fill
        bg = torch.zeros_like(image)
        image = torch.where(mask.unsqueeze(0) == 1, image, bg)

        # 8) Mild contrast boost
        cvar = random.uniform(1.0, 1.2)
        image = F.adjust_contrast(image, cvar).clamp(0, 1)

        # 9) 2D rotations/flips
        rot = random.uniform(-10, 10)
        image = F.rotate(image, rot, interpolation=F.InterpolationMode.BILINEAR, fill=(0,0,0))
        mask = F.rotate(mask.unsqueeze(0), rot, interpolation=F.InterpolationMode.NEAREST, fill=0)[0].to(torch.uint8)
        if random.random() < 0.5:
            image = F.hflip(image); mask = F.hflip(mask)
        if random.random() < 0.5:
            image = F.vflip(image); mask = F.vflip(mask)

        # 10) Bounding box & target
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
