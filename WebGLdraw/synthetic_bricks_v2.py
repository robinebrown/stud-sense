import os
import math
import random
import torch
import re
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

class SyntheticBrickDatasetV2(Dataset):
    """
    Standalone grayscale brick render dataset.
    Renders each LEGO part in constant light gray (0.8) under a headlight,
    with multiple random views, tight crops, and adjustable render resolution.

    Parameters:
        obj_dir (str): Path to .obj file or directory of .obj files.
        image_size (int): Final output image size (square H=W).
        render_scale (int): Supersampling factor for rendering. render_size = image_size * render_scale.
        views_per_obj (int): How many random views to render per part.
        device (str): Torch device ("cuda" or "cpu").
        camera_scale (float): Multiplier for camera distance.
        fov (float): Field of view for perspective camera.
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
        # Device & render sizes
        self.device = torch.device(device)
        self.image_size = image_size
        self.render_size = image_size * render_scale
        self.views_per_obj = views_per_obj
        self.camera_scale = camera_scale
        self.fov = fov

        # 1) Collect OBJ paths
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

        # 2) Load meshes & compute radii
        self.meshes = load_objs_as_meshes(self.mesh_paths, device=self.device)
        self.radii = []
        for verts in self.meshes.verts_list():
            center = verts.mean(0, keepdim=True)
            self.radii.append(((verts - center).norm(dim=1)).max().item())

        # 3) Apply constant light gray textures
        gray = 0.85
        feats = [torch.full((v.shape[0], 3), gray, device=self.device) for v in self.meshes.verts_list()]
        self.meshes.textures = TexturesVertex(verts_features=feats)

        # 4) Renderer setup: headlight only
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
        # total = parts × views per part
        return len(self.mesh_paths) * self.views_per_obj

    def __getitem__(self, idx):
        # Determine mesh index
        mesh_idx = idx % len(self.mesh_paths)
        mesh = self.meshes[mesh_idx]
        rad  = self.radii[mesh_idx]

        # 1) Random camera pose
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

        # 2) Render & mask
        out = self.renderer(mesh, R=R, T=T)
        rgb = out[0, ..., :3]
        frags = self.renderer.rasterizer(mesh, R=R, T=T)
        mask = (frags.pix_to_face[..., 0] >= 0).squeeze(0).to(torch.uint8)

        # 3) Composite gray over black
        comp = rgb * mask.unsqueeze(-1).float()

        # 4) Crop to bounds + margin
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

        # 5) Pad to square
        c, h, w = crop.shape
        if h > w:
            diff = h - w
            pad = (diff // 2, 0, diff - diff // 2, 0)
        else:
            diff = w - h
            pad = (0, diff // 2, 0, diff - diff // 2)
        crop = F.pad(crop, pad, fill=0)
        mask_crop = F.pad(mask_crop.unsqueeze(0), pad, fill=0).squeeze(0)

        # 6) Downsample to final size
        image = F.resize(crop, [self.image_size, self.image_size])
        mask_out = F.resize(
            mask_crop.unsqueeze(0).float(),
            [self.image_size, self.image_size],
            interpolation=F.InterpolationMode.NEAREST
        )[0].to(torch.uint8)

        return image, {'mask': mask_out, 'is_transparent': 0.0}
