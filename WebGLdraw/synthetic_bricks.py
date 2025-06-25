# synthetic_bricks.py

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
      - Random camera azimuth [0–360°], elevation [−75°…+75°]
      - Crop→pad→resize without distortion
      - Mild contrast boost, 2D rotations/flips
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

        # 2) Load meshes to GPU
        meshes = load_objs_as_meshes(paths, device=self.device)
        valid = [i for i, v in enumerate(meshes.verts_list()) if v.shape[0] > 0]
        self.meshes = meshes[valid]

        # 3) Radii for camera distance
        self.radii = []
        for verts in self.meshes.verts_list():
            center = verts.mean(0, keepdim=True)
            self.radii.append((verts - center).norm(dim=1).max().item())

        # 4) White textures
        feats = [torch.ones((v.shape[0], 3), device=self.device)
                 for v in self.meshes.verts_list()]
        self.meshes.textures = TexturesVertex(verts_features=feats)

        # 5) Renderer setup
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
            direction=[[0.0, 0.0, -1.0]],
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
        # 1) Possibly tint
        mesh = self.meshes[idx]
        if random.random() < 0.45:
            h, s, v = random.random(), random.uniform(0.5,1.0), random.uniform(0.6,1.0)
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            verts = mesh.verts_list()[0]
            cfeat = torch.tensor([r,g,b], device=self.device)
            feats = cfeat.unsqueeze(0).repeat(verts.shape[0],1)
            mesh.textures = TexturesVertex(verts_features=[feats])

        # 2) Random camera
        radius = self.radii[idx]
        half_fov = math.radians(self.fov/2)
        dist = (radius*self.camera_scale)/math.tan(half_fov)
        azim, elev = random.uniform(0,360), random.uniform(-75,75)
        R, T = look_at_view_transform(dist, elev, azim, device=self.device)
        cam = FoVPerspectiveCameras(device=self.device, fov=self.fov, R=R, T=T)
        self.renderer.rasterizer.cameras = cam
        self.renderer.shader.cameras    = cam

        # 3) Headlight align (optional)
        cam_to_world = R[0].T
        dir_world = (torch.tensor([0,0,-1], device=self.device).view(1,3) @ cam_to_world).cpu().tolist()
        lights = DirectionalLights(device=self.device, direction=dir_world,
                                   ambient_color=[[0.1]*3], diffuse_color=[[0.8]*3],
                                   specular_color=[[0.3]*3])
        self.renderer.shader.lights = lights

        # 4) Render
        out   = self.renderer(mesh, R=R, T=T)
        rgb   = out[0,...,:3]
        frags = self.renderer.rasterizer(mesh, R=R, T=T)
        mask  = (frags.pix_to_face[...,0]>=0).squeeze(0)

        # 5) To tensors
        image = rgb.permute(2,0,1)   # (3,H,W)
        mask  = mask.to(torch.uint8)

        # 6) Crop/pad/resize
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel():
            m = 16
            y0, y1 = max(0,ys.min()-m), min(mask.shape[0],ys.max()+m)
            x0, x1 = max(0,xs.min()-m), min(mask.shape[1],xs.max()+m)
        else:
            y0,y1,x0,x1 = 0,1,0,1
        crop_i = F.resize(F.pad(F.crop(image, y0,x0,y1-y0,x1-x0),
                                self._pad((y1-y0,x1-x0)), fill=0),
                          [self.image_size]*2)
        crop_m = F.resize(F.pad(F.crop(mask.unsqueeze(0),y0,x0,y1-y0,x1-x0),
                                self._pad((y1-y0,x1-x0)), fill=0).squeeze(0),
                          [self.image_size]*2, interpolation=F.InterpolationMode.NEAREST)
        image, mask = crop_i.clamp(0,1), crop_m.to(torch.uint8)

        # 7) Contrast & 2D augs...
        image = F.adjust_contrast(image, random.uniform(1.0,1.2)).clamp(0,1)
        rot = random.uniform(-10,10)
        image = F.rotate(image, rot, fill=(0,0,0))
        mask  = F.rotate(mask.unsqueeze(0), rot, interpolation=F.InterpolationMode.NEAREST,
                         fill=0)[0].to(torch.uint8)
        if random.random()<0.5:
            image,mask = F.hflip(image), F.hflip(mask)
        if random.random()<0.5:
            image,mask = F.vflip(image), F.vflip(mask)

        # 8) Build target
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel():
            y0,y1,x0,x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
        else:
            y0,y1,x0,x1 = 0,1,0,1
        box = torch.tensor([[x0,y0,x1,y1]], dtype=torch.float32)
        target = {
            "boxes":    box,
            "labels":   torch.tensor([idx+1], dtype=torch.int64),
            "masks":    mask.unsqueeze(0),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area":     (x1-x0)*(y1-y0),
            "iscrowd":  torch.zeros(1, dtype=torch.int64),
        }

        # ——— MOVE TO CPU FOR PINNING ———
        image = image.cpu()
        for k,v in list(target.items()):
            if isinstance(v, torch.Tensor):
                target[k] = v.cpu()

        return image, target
