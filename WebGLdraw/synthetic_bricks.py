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
    Grayscale renders (with optional color variation) of LEGO parts:
      - White or randomly tinted piece on black background
      - Headlight directional lighting
      - Random azimuth [0–360°], elevation [−75°…+75°]
      - Crop→pad→resize to square
      - Mild contrast boost, small 2D rotations/flips
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

        # 1) Gather .obj file paths
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

        # 2) Load meshes onto GPU
        meshes = load_objs_as_meshes(paths, device=self.device)
        valid = [i for i, v in enumerate(meshes.verts_list()) if v.shape[0] > 0]
        self.meshes = meshes[valid]

        # 3) Compute each mesh’s bounding radius
        self.radii = []
        for verts in self.meshes.verts_list():
            center = verts.mean(0, keepdim=True)
            self.radii.append((verts - center).norm(dim=1).max().item())

        # 4) Initialize white vertex textures
        feats = [torch.ones((v.shape[0], 3), device=self.device)
                 for v in self.meshes.verts_list()]
        self.meshes.textures = TexturesVertex(verts_features=feats)

        # 5) Set up a headless MeshRenderer placeholder
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
        mesh = self.meshes[idx]

        # 1) 45% chance to tint the mesh
        if random.random() < 0.45:
            h, s, v = random.random(), random.uniform(0.5,1.0), random.uniform(0.6,1.0)
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            verts = mesh.verts_list()[0]
            col = torch.tensor([r, g, b], device=self.device, dtype=torch.float32)
            feats = col.unsqueeze(0).repeat(verts.shape[0], 1)
            mesh.textures = TexturesVertex(verts_features=[feats])

        # 2) Random camera pose
        radius = self.radii[idx]
        half_fov = math.radians(self.fov / 2)
        dist = (radius * self.camera_scale) / math.tan(half_fov)
        azim = random.uniform(0, 360)
        elev = random.uniform(-75, 75)
        R, T = look_at_view_transform(dist, elev, azim, device=self.device)
        cam = FoVPerspectiveCameras(device=self.device, fov=self.fov, R=R, T=T)
        self.renderer.rasterizer.cameras = cam
        self.renderer.shader.cameras    = cam

        # 3) Headlight aligned with camera forward
        cam_to_world = R[0].T
        dir_cam = torch.tensor([0.0, 0.0, -1.0],
                               device=self.device,
                               dtype=cam_to_world.dtype).view(1, 3)
        dir_world = (dir_cam @ cam_to_world).cpu().tolist()
        lights = DirectionalLights(
            device=self.device,
            direction=dir_world,
            ambient_color=[[0.1,0.1,0.1]],
            diffuse_color=[[0.8,0.8,0.8]],
            specular_color=[[0.3,0.3,0.3]]
        )
        self.renderer.shader.lights = lights

        # 4) Render RGB + mask
        out   = self.renderer(mesh, R=R, T=T)
        rgb   = out[0,...,:3]
        frags = self.renderer.rasterizer(mesh, R=R, T=T)
        mask  = (frags.pix_to_face[...,0] >= 0).squeeze(0)

        # 5) To tensors
        image = rgb.permute(2,0,1)         # (3,H,W)
        mask  = mask.to(torch.uint8)       # (H,W)

        # 6) Crop to object, pad to square, resize
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

        crop_img  = F.crop(image, y0, x0, y1-y0, x1-x0)
        crop_mask = F.crop(mask.unsqueeze(0), y0, x0, y1-y0, x1-x0).squeeze(0)
        ch, cw = crop_img.shape[1:]
        if ch > cw:
            diff = ch - cw
            pad = (diff//2, 0, diff - diff//2, 0)
        else:
            diff = cw - ch
            pad = (0, diff//2, 0, diff - diff//2)
        crop_img  = F.pad(crop_img, pad, fill=0)
        crop_mask = F.pad(crop_mask.unsqueeze(0), pad, fill=0).squeeze(0)
        image = F.resize(crop_img,  [self.image_size]*2, interpolation=F.InterpolationMode.BILINEAR)
        mask  = F.resize(crop_mask.unsqueeze(0).float(),
                         [self.image_size]*2,
                         interpolation=F.InterpolationMode.NEAREST)[0].to(torch.uint8)

        # 7) Fill black background & contrast
        bg = torch.zeros_like(image)
        image = torch.where(mask.unsqueeze(0)==1, image, bg)
        image = F.adjust_contrast(image, random.uniform(1.0,1.2)).clamp(0,1)

        # 8) 2D rotations/flips
        rot = random.uniform(-10,10)
        image = F.rotate(image, rot, fill=(0,0,0))
        mask  = F.rotate(mask.unsqueeze(0), rot,
                         interpolation=F.InterpolationMode.NEAREST,
                         fill=0)[0].to(torch.uint8)
        if random.random()<0.5:
            image, mask = F.hflip(image), F.hflip(mask)
        if random.random()<0.5:
            image, mask = F.vflip(image), F.vflip(mask)

        # 9) Build target dict
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel():
            y0, y1 = ys.min().item(), ys.max().item()+1
            x0, x1 = xs.min().item(), xs.max().item()+1
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
        for k, v in list(target.items()):
            if isinstance(v, torch.Tensor):
                target[k] = v.cpu()

        return image, target
