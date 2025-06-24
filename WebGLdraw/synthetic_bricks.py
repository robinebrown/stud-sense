import os
import random
import math
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
    PointLights,
    TexturesVertex
)
from torchvision.transforms import functional as F

class SyntheticBrickDataset(Dataset):
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

        # 1) Load exactly one OBJ if path is a file, else scan folder
        if os.path.isfile(obj_dir) and obj_dir.lower().endswith(".obj"):
            paths = [obj_dir]
        else:
            paths = [
                os.path.join(obj_dir, f)
                for f in os.listdir(obj_dir)
                if f.lower().endswith(".obj")
            ]
        if max_meshes:
            paths = paths[:max_meshes]
        print(f"→ Found {len(paths)} .obj files in {obj_dir}")

        # 2) Load meshes
        meshes = load_objs_as_meshes(paths, device=self.device)
        valid = [i for i,v in enumerate(meshes.verts_list()) if v.shape[0]>0]
        self.meshes = meshes[valid]

        # 3) Precompute bounding radii
        self.radii = []
        for verts in self.meshes.verts_list():
            c = verts.mean(0, keepdim=True)
            self.radii.append((verts - c).norm(dim=1).max().item())

        # 4) White vertex textures
        feats = [torch.ones((v.shape[0],3),device=self.device)
                 for v in self.meshes.verts_list()]
        self.meshes.textures = TexturesVertex(verts_features=feats)

        # 5) Build renderer (placeholder camera)
        placeholder_cam = FoVPerspectiveCameras(device=self.device, fov=self.fov)
        rast_settings = RasterizationSettings(
            image_size=self.render_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
            max_faces_per_bin=1000
        )
        lights = PointLights(device=self.device, location=[[0,0,0]])
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=placeholder_cam, raster_settings=rast_settings),
            shader=SoftPhongShader(device=self.device, cameras=placeholder_cam, lights=lights)
        ).to(self.device)

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):
        mesh   = self.meshes[idx]
        radius = self.radii[idx]

        # Try up to max_tries to get a visible view
        for _ in range(self.max_tries):
            # 1) Random spherical coords
            azim = random.uniform(0, 360.0)
            elev = random.uniform(-30.0, 60.0)
            # 2) Compute distance so brick fits in view
            half_fov = math.radians(self.fov/2)
            dist = (radius * self.camera_scale) / math.tan(half_fov)
            # 3) Get R, T via look_at helper
            R, T = look_at_view_transform(dist, elev, azim, device=self.device)
            # 4) Build camera and set it
            cam = FoVPerspectiveCameras(device=self.device, fov=self.fov, R=R, T=T)
            self.renderer.rasterizer.cameras = cam
            self.renderer.shader.cameras    = cam
            # 5) Render
            rgb   = self.renderer(mesh, R=R, T=T)[0,...,:3]
            frags = self.renderer.rasterizer(mesh, R=R, T=T)
            mask  = (frags.pix_to_face[...,0]>=0).squeeze(0)
            if mask.any():
                break
        else:
            # Guaranteed fallback: use a fixed view
            R, T = look_at_view_transform(dist, 30.0, 45.0, device=self.device)
            cam = FoVPerspectiveCameras(device=self.device, fov=self.fov, R=R, T=T)
            self.renderer.rasterizer.cameras = cam
            self.renderer.shader.cameras    = cam
            rgb   = self.renderer(mesh, R=R, T=T)[0,...,:3]
            frags = self.renderer.rasterizer(mesh, R=R, T=T)
            mask  = (frags.pix_to_face[...,0]>=0).squeeze(0)

        # Convert to tensors
        image = rgb.permute(2,0,1)       # (3, H, W)
        mask_t = mask.to(torch.uint8)    # (H, W)

        # Crop & downsample at high res
        H, W = mask_t.shape
        ys, xs = torch.nonzero(mask_t, as_tuple=True)
        if ys.numel()==0:
            y0,x0,y1,x1 = 0,0,1,1
        else:
            m = 16
            y0 = max(0, ys.min().item()-m)
            y1 = min(H, ys.max().item()+m)
            x0 = max(0, xs.min().item()-m)
            x1 = min(W, xs.max().item()+m)
        crop_img  = F.crop(image, y0, x0, y1-y0, x1-x0)
        crop_mask = F.crop(mask_t.unsqueeze(0), y0, x0, y1-y0, x1-x0).squeeze(0)
        image = F.resize(crop_img, [self.image_size, self.image_size], interpolation=F.InterpolationMode.BILINEAR)
        mask  = F.resize(crop_mask.unsqueeze(0).float(), [self.image_size, self.image_size], interpolation=F.InterpolationMode.NEAREST)[0].to(torch.uint8)

        # Compute tight bbox
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel()==0:
            y0,x0,y1,x1 = 0,0,1,1
        else:
            y0,y1 = ys.min().item(), ys.max().item()+1
            x0,x1 = xs.min().item(), xs.max().item()+1

        box = torch.tensor([[x0, y0, x1, y1]], dtype=torch.float32, device=self.device)
        target = {
            "boxes":    box,
            "labels":   torch.tensor([idx+1], dtype=torch.int64, device=self.device),
            "masks":    mask.unsqueeze(0),
            "image_id": torch.tensor([idx], device=self.device),
            "area":     (x1-x0)*(y1-y0),
            "iscrowd":  torch.zeros(1, dtype=torch.int64, device=self.device)
        }
        return image, target
