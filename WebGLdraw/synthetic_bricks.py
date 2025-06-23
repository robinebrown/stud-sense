import os, random, torch
from torch.utils.data import Dataset
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings,
    MeshRenderer, MeshRasterizer, SoftPhongShader,
    PointLights, TexturesVertex
)

def random_spherical_pose(device):
    theta = torch.rand(1) * 2 * torch.pi
    phi   = torch.acos(2 * torch.rand(1) - 1)
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    R = torch.eye(3, device=device).unsqueeze(0)
    T = torch.stack([x, y, z], dim=1)
    return R, T

class SyntheticBrickDataset(Dataset):
    def __init__(self, obj_dir, image_size=256, max_tries=5, smoke_test=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # gather .obj paths
        all_paths = sorted([
            os.path.join(obj_dir, f)
            for f in os.listdir(obj_dir)
            if f.lower().endswith('.obj')
        ])
        if smoke_test:
            print(f"→ [Smoke-test] using first 200 of {len(all_paths)} meshes")
            all_paths = all_paths[:200]
        self.obj_paths = all_paths

        print(f"→ Loading {len(self.obj_paths)} meshes onto {self.device} …")
        meshes = load_objs_as_meshes(self.obj_paths, device=self.device)

        # filter out empties
        verts_list = meshes.verts_list()
        valid_idx = [i for i,v in enumerate(verts_list) if v.shape[0]>0]
        if len(valid_idx) < len(verts_list):
            print(f"   · filtered out {len(verts_list)-len(valid_idx)} empty meshes")
        self.meshes = meshes[valid_idx]
        self.obj_paths = [self.obj_paths[i] for i in valid_idx]

        # white vertex colors
        verts_list = self.meshes.verts_list()
        verts_features = [
            torch.ones((v.shape[0],3), device=self.device)
            for v in verts_list
        ]
        self.meshes.textures = TexturesVertex(verts_features=verts_features)

        # renderer
        cameras = FoVPerspectiveCameras(device=self.device)
        raster_settings = RasterizationSettings(
            image_size=image_size, blur_radius=0.0, faces_per_pixel=1
        )
        lights = PointLights(device=self.device, location=[[0.0,0.0,3.0]])
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, cameras=cameras, lights=lights
            )
        )

        self.max_tries = max_tries

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):
        mesh = self.meshes[idx]
        # random views until visible
        for _ in range(self.max_tries):
            R, T = random_spherical_pose(self.device)
            rgb = self.renderer(mesh, R=R, T=T)[0,...,:3]
            frags = self.renderer.rasterizer(mesh, R=R, T=T)
            mask = (frags.pix_to_face[...,0] >= 0).squeeze(0)
            if mask.any():
                break
        else:
            # fallback front view
            R = torch.eye(3, device=self.device).unsqueeze(0)
            T = torch.tensor([[0,0,3]], dtype=torch.float32, device=self.device)
            rgb = self.renderer(mesh, R=R, T=T)[0,...,:3]
            frags = self.renderer.rasterizer(mesh, R=R, T=T)
            mask = (frags.pix_to_face[...,0] >= 0).squeeze(0)

        # image and mask
        image = rgb.permute(2,0,1)           # (3,H,W)
        mask  = mask.to(torch.uint8)        # (H,W)

        # bounding box (ensure positive area)
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel()==0 or xs.numel()==0:
            ymin, ymax, xmin, xmax = 0, 1, 0, 1
        else:
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()
            if xmax <= xmin:
                xmax = xmin + 1
            if ymax <= ymin:
                ymax = ymin + 1

        box = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)

        target = {
            "boxes":    box,
            "labels":   torch.tensor([idx+1], dtype=torch.int64),
            "masks":    mask.unsqueeze(0),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area":     (xmax-xmin)*(ymax-ymin),
            "iscrowd":  torch.zeros(1, dtype=torch.int64)
        }
        return image, target
