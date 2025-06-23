# synthetic_bricks.py
import os, random, torch
from torch.utils.data import Dataset
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings,
    MeshRenderer, MeshRasterizer, SoftPhongShader,
    PointLights, TexturesVertex
)

def random_spherical_pose(device):
    theta = torch.rand(1, device=device) * 2 * torch.pi
    phi   = torch.acos(2 * torch.rand(1, device=device) - 1)
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    R = torch.eye(3, device=device).unsqueeze(0)
    T = torch.stack([x, y, z], dim=1)
    return R, T

class SyntheticBrickDataset(Dataset):
    def __init__(self, obj_dir, image_size=256, max_tries=5, max_meshes=None, device="cuda"):
        self.device = torch.device(device)

        # Gather .obj file paths
        all_paths = [os.path.join(obj_dir, f)
                     for f in os.listdir(obj_dir)
                     if f.lower().endswith('.obj')]
        if max_meshes is not None:
            all_paths = all_paths[:max_meshes]
        print(f"→ Found {len(all_paths)} .obj files in {obj_dir}")

        # Load meshes onto GPU
        meshes = load_objs_as_meshes(all_paths, device=self.device)
        verts_list = meshes.verts_list()
        valid_idx = [i for i, v in enumerate(verts_list) if v.shape[0] > 0]
        if len(valid_idx) < len(verts_list):
            print(f"   · filtered out {len(verts_list)-len(valid_idx)} empty meshes")
        self.meshes = meshes[valid_idx]
        self.obj_paths = [all_paths[i] for i in valid_idx]

        # White vertex textures
        verts_list = self.meshes.verts_list()
        verts_features = [torch.ones((v.shape[0], 3), device=self.device)
                          for v in verts_list]
        self.meshes.textures = TexturesVertex(verts_features=verts_features)

        # GPU-side renderer
        cameras = FoVPerspectiveCameras(device=self.device)
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,             # auto-tuned
            max_faces_per_bin=1000  # leverage H100’s memory
        )
        lights = PointLights(device=self.device, location=[[0.0, 0.0, 3.0]])
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cameras,
                lights=lights
            )
        )

        self.max_tries = max_tries

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):
        mesh = self.meshes[idx]
        # random-view until visible
        for _ in range(self.max_tries):
            R, T = random_spherical_pose(self.device)
            rgb = self.renderer(mesh, R=R, T=T)[0, ..., :3]
            frags = self.renderer.rasterizer(mesh, R=R, T=T)
            mask = frags.pix_to_face[..., 0] >= 0
            if mask.any():
                break
        else:
            R = torch.eye(3, device=self.device).unsqueeze(0)
            T = torch.tensor([[0, 0, 3]], device=self.device, dtype=torch.float32)
            rgb = self.renderer(mesh, R=R, T=T)[0, ..., :3]
            frags = self.renderer.rasterizer(mesh, R=R, T=T)
            mask = frags.pix_to_face[..., 0] >= 0

        # prepare output
        image = rgb.permute(2, 0, 1)
        mask = mask.squeeze(0).to(torch.uint8)
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel() == 0:
            ymin, ymax, xmin, xmax = 0, 1, 0, 1
        else:
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()
            xmax = max(xmax, xmin + 1)
            ymax = max(ymax, ymin + 1)
        box = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32, device=self.device)
        target = {
            "boxes":    box,
            "labels":   torch.tensor([idx + 1], dtype=torch.int64, device=self.device),
            "masks":    mask.unsqueeze(0),
            "image_id": torch.tensor([idx], device=self.device),
            "area":     (xmax - xmin) * (ymax - ymin),
            "iscrowd":  torch.zeros(1, dtype=torch.int64, device=self.device)
        }
        return image, target
