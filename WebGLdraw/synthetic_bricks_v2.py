import os
import random
import torch
import csv
import re
from torch.utils.data import Dataset
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    SoftPhongShader, DirectionalLights, TexturesVertex,
    BlendParams
)
from torchvision.transforms import functional as F

class SyntheticBrickDatasetV2(Dataset):
    def __init__(
        self,
        obj_dir,
        image_size=256,
        max_meshes=None,
        device="cuda",
        views_per_obj=1,
        colors_csv="colors.csv",
        elements_csv="elements.csv",
        camera_scale=2.5
    ):
        # Device and sizing
        self.device = torch.device(device)
        self.image_size = image_size
        self.render_size = image_size * 4  # high-res render for crisp downsampling
        self.views_per_obj = views_per_obj
        self.camera_scale = camera_scale

        # 1) Load palette RGB
        self.color_info = {}
        with open(colors_csv) as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            use_rgb = all(k in fields for k in ('r','g','b'))
            use_hex = 'rgb' in fields
            for row in reader:
                cid = int(row['id'])
                if use_rgb:
                    rgb = (int(row['r'])/255.0,
                           int(row['g'])/255.0,
                           int(row['b'])/255.0)
                elif use_hex:
                    h = row['rgb'].lstrip('#')
                    rgb = (int(h[0:2],16)/255.0,
                           int(h[2:4],16)/255.0,
                           int(h[4:6],16)/255.0)
                else:
                    rgb = (1.0, 1.0, 1.0)
                self.color_info[cid] = rgb

        # 2) Load part -> (cid, is_trans) mappings
        self.part_colours = {}
        with open(elements_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row['part_num']
                cid = int(row['color_id'])
                is_trans = row.get('is_trans','0') in ('1','True','true')
                self.part_colours.setdefault(pid, []).append((cid, is_trans))

        # 3) Collect and filter .obj files
        if os.path.isfile(obj_dir) and obj_dir.endswith('.obj'):
            paths = [obj_dir]
        else:
            paths = [os.path.join(obj_dir, fn)
                     for fn in os.listdir(obj_dir)
                     if fn.endswith('.obj')]
        filtered = []
        for p in paths:
            pid = os.path.splitext(os.path.basename(p))[0]
            if re.match(r'^\d+[A-Za-z]?$', pid):
                filtered.append(p)
            else:
                print(f"→ Skipping non-part obj: {pid}")
        if max_meshes:
            filtered = filtered[:max_meshes]
        print(f"→ Using {len(filtered)} .obj files from {obj_dir}")
        self.mesh_paths = filtered

        # 4) Load meshes and compute radii
        self.meshes = load_objs_as_meshes(self.mesh_paths, device=self.device)
        self.radii = []
        for mesh in self.meshes:
            verts = mesh.verts_list()[0]
            center = verts.mean(dim=0, keepdim=True)
            rad = ((verts - center).norm(dim=1)).max().item()
            self.radii.append(rad)

        # 5) Renderer with black background
        cams = FoVPerspectiveCameras(device=self.device, fov=60.0)
        rast = RasterizationSettings(
            image_size=self.render_size,
            blur_radius=0.0,
            faces_per_pixel=1
        )
        self.renderer = MeshRenderer(
            MeshRasterizer(cameras=cams, raster_settings=rast),
            SoftPhongShader(
                device=self.device,
                cameras=cams,
                lights=DirectionalLights(device=self.device, direction=[[0,0,-1]]),
                blend_params=BlendParams(background_color=(0.0, 0.0, 0.0))
            )
        )

    def __len__(self):
        return len(self.meshes) * self.views_per_obj

    def __getitem__(self, idx):
        mesh = self.meshes[idx % len(self.meshes)]
        pid  = os.path.splitext(os.path.basename(self.mesh_paths[idx % len(self.mesh_paths)]))[0]
        rad  = self.radii[idx % len(self.radii)]

        # 6) Sample color or fallback
        variants = self.part_colours.get(pid)
        if variants:
            cid, is_trans = random.choice(variants)
            rgb = self.color_info.get(cid, (0.5,0.5,0.5))
            alpha = 0.5 if is_trans else 1.0
        else:
            rgb, alpha = (0.5,0.5,0.5), 1.0  # placeholder gray

        # 7) Apply vertex colors
        verts = mesh.verts_list()[0]
        feats = torch.tensor(rgb, device=self.device, dtype=torch.float32)
        feats = feats.view(1,3).expand(verts.shape[0],3)
        mesh.textures = TexturesVertex(verts_features=[feats])

        # 8) Camera transform
        half_fov = torch.deg2rad(torch.tensor(60.0/2, device=self.device, dtype=torch.float32))
        dist = rad / torch.tan(half_fov) * self.camera_scale
        R, T = look_at_view_transform(
            dist=dist.item(), elev=random.uniform(-75,75), azim=random.uniform(0,360), device=self.device
        )
        cam = FoVPerspectiveCameras(device=self.device, fov=60.0, R=R, T=T)
        self.renderer.rasterizer.cameras = cam
        self.renderer.shader.cameras    = cam
        # correct dtype for light direction
        light_dir = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).view(1,3)
        dir_world = (light_dir @ R[0].T).tolist()
        self.renderer.shader.lights = DirectionalLights(device=self.device, direction=dir_world)

        # 9) Render + mask
        rendered = self.renderer(mesh, R=R, T=T)[0,...,:3]
        frags    = self.renderer.rasterizer(mesh, R=R, T=T)
        mask     = (frags.pix_to_face[...,0] >= 0).squeeze(0).to(torch.uint8)

        # 10) Composite with alpha over black
        comp = rendered * alpha

        # 11) Tight crop + 16px margin
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel():
            m = 16
            y0 = max(0, ys.min().item() - m); y1 = min(mask.shape[0], ys.max().item() + m)
            x0 = max(0, xs.min().item() - m); x1 = min(mask.shape[1], xs.max().item() + m)
        else:
            y0,y1,x0,x1 = 0,1,0,1
        crop      = comp[y0:y1, x0:x1, :].permute(2,0,1)
        mask_crop = mask[y0:y1, x0:x1]

        # 12) Pad to square
        c,h,w = crop.shape
        if h > w:
            diff = h-w; pad = (diff//2,0,diff-diff//2,0)
        else:
            diff = w-h; pad = (0,diff//2,0,diff-diff//2)
        crop      = F.pad(crop, pad, fill=0)
        mask_crop = F.pad(mask_crop.unsqueeze(0), pad, fill=0).squeeze(0)

        # 13) Downsample to target size
        image    = F.resize(crop,    [self.image_size, self.image_size])
        mask_aug = F.resize(
            mask_crop.unsqueeze(0).float(),
            [self.image_size, self.image_size],
            interpolation=F.InterpolationMode.NEAREST
        )[0].to(torch.uint8)

        return image, {'mask': mask_aug, 'is_transparent': float(alpha < 1.0)}
