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
from pytorch3d.structures import join_meshes_as_batch
from torchvision.transforms.v2 import functional as F
from torchvision.io import write_png

# Default list of 20 prototype part IDs
PART_IDS = [
    "3001","3003","3020","3022","3040","54200","3665","3062","4740","15254",
    "4865","32000","2780","3626","2555","6238","51239","88293","2817","30414"
]

class SyntheticBrickProtoDataset(Dataset):
    """
    Dataset holding meshes, centers, radii, renderer, and utilities.
    Rendering is batched in the main loop for speed.
    """
    def __init__(self,
                 obj_dir,
                 part_ids,
                 image_size=330,
                 render_scale=3,
                 views_per_obj=5,
                 device="cpu",
                 camera_scale=2.5,
                 fov=60.0):
        # World up: OBJs are often Z-up; change to Y-up if needed
        UP_Z = ((0.0, 0.0, 1.0),)

        self.device = torch.device(device)
        self.image_size = image_size
        self.render_size = image_size * render_scale
        self.views_per_obj = views_per_obj
        self.camera_scale = camera_scale
        self.fov = fov
        self.up = UP_Z

        # Build full paths to each part's .obj
        self.mesh_paths = [os.path.join(obj_dir, f"{pid}.obj") for pid in part_ids]
        print(f"Prototype parts: {len(self.mesh_paths)} meshes from {obj_dir}")

        # Load meshes on device
        self.meshes = load_objs_as_meshes(self.mesh_paths, device=self.device)

        # Precompute centers and radii
        self.centers = []
        self.radii = []
        for verts in self.meshes.verts_list():
            center = verts.mean(0, keepdim=True)
            self.centers.append(center.squeeze(0))  # [3]
            self.radii.append(((verts - center).norm(dim=1)).max().item())

        # Assign constant light gray vertex color
        gray = 0.8
        feats = [torch.full((v.shape[0], 3), gray, device=self.device)
                 for v in self.meshes.verts_list()]
        self.meshes.textures = TexturesVertex(verts_features=feats)

        # Renderer
        self.base_cameras = FoVPerspectiveCameras(device=self.device, fov=self.fov)
        rast = RasterizationSettings(
            image_size=self.render_size,
            blur_radius=0.0,
            faces_per_pixel=1
        )
        # Keep lights object; we will update its direction per batch
        self.lights = DirectionalLights(
            device=self.device,
            direction=[[0.0, 0.0, -1.0]],
            ambient_color=[[0.1, 0.1, 0.1]],
            diffuse_color=[[0.8, 0.8, 0.8]],
            specular_color=[[0.3, 0.3, 0.3]]
        )
        self.renderer = MeshRenderer(
            MeshRasterizer(cameras=self.base_cameras, raster_settings=rast),
            SoftPhongShader(device=self.device, cameras=self.base_cameras, lights=self.lights)
        )

    def __len__(self):
        return len(self.mesh_paths) * self.views_per_obj

    # ---- Batched utilities ----

    def sample_azim_elev(self, n):
        # Avoid poles to reduce roll ambiguity
        elev = torch.empty(n, device=self.device).uniform_(-75.0, 75.0)
        azim = torch.empty(n, device=self.device).uniform_(0.0, 360.0)
        return azim, elev

    def distance_for_mesh(self, mesh_idx):
        half = math.radians(self.fov / 2)
        dist = self.radii[mesh_idx] * self.camera_scale / math.tan(half)
        return float(dist)

    def render_batch_views(self, mesh_idx, azim_deg, elev_deg):
        """
        Render a batch of views for a single mesh index.
        azim_deg, elev_deg: 1D tensors on device with same length N.
        Returns:
          imgs: [N, H, W, 3] float tensor in [0,1] on device
          masks: [N, H, W] uint8 tensor (0/1) on device
        """
        mesh = self.meshes[mesh_idx]
        center = self.centers[mesh_idx].unsqueeze(0)  # [1,3]
        N = azim_deg.shape[0]

        # Dist can be scalar; broadcast by PyTorch3D
        dist = self.distance_for_mesh(mesh_idx)

        # Cameras for batch
        R, T = look_at_view_transform(
            dist=dist,
            elev=elev_deg,
            azim=azim_deg,
            at=center.repeat(N, 1),
            up=self.up,
            degrees=True,
            device=self.device
        )
        cameras = FoVPerspectiveCameras(device=self.device, fov=self.fov, R=R, T=T)
        # Update renderer cameras
        self.renderer.rasterizer.cameras = cameras
        self.renderer.shader.cameras = cameras

        # Headlight alignment (batched): camera forward (0,0,-1) in cam → world
        # cam_to_world = R^T; for batch we can multiply by vector
        cam_to_world = R.transpose(1, 2)  # [N,3,3]
        dir_cam = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=cam_to_world.dtype).view(1, 3, 1)
        world_dir = torch.bmm(cam_to_world, dir_cam).squeeze(-1)  # [N,3]
        # Normalize (defensive)
        world_dir = torch.nn.functional.normalize(world_dir, dim=1)
        # Update lights direction (batched)
        self.lights.direction = world_dir

        # Duplicate mesh N times into a batch
        mesh_batch = join_meshes_as_batch([mesh] * N)
        # Render
        out = self.renderer(mesh_batch, R=R, T=T)[..., :3]  # [N,H,W,3]

        # Fragments for mask
        frags = self.renderer.rasterizer(mesh_batch, R=R, T=T)
        mask = (frags.pix_to_face[..., 0] >= 0).to(torch.uint8)  # [N,H,W]

        return out, mask

    @staticmethod
    def _crop_square_with_border(img_chw, mask_hw, border_px=45):
        """
        Per-sample crop to tight bbox, pad to square, then add border.
        img_chw: [3,H,W] float
        mask_hw: [H,W] uint8
        Returns [3,H',W'], [H',W'] on same device.
        """
        ys, xs = torch.nonzero(mask_hw, as_tuple=True)
        H, W = mask_hw.shape
        if ys.numel():
            m = 16
            y0 = max(0, int(ys.min().item()) - m)
            y1 = min(H, int(ys.max().item()) + m)
            x0 = max(0, int(xs.min().item()) - m)
            x1 = min(W, int(xs.max().item()) + m)
        else:
            y0, y1, x0, x1 = 0, 1, 0, 1

        crop = img_chw[:, y0:y1, x0:x1]
        mask_crop = mask_hw[y0:y1, x0:x1]

        # Pad to square
        _, h, w = crop.shape
        if h > w:
            diff = h - w
            pad = (diff // 2, 0, diff - diff // 2, 0)  # (left, right, top, bottom)
        else:
            diff = w - h
            pad = (0, diff // 2, 0, diff - diff // 2)

        crop = F.pad(crop, pad, fill=0)
        mask_crop = F.pad(mask_crop.unsqueeze(0), pad, fill=0).squeeze(0)

        # Extra border
        b = border_px
        crop = F.pad(crop, (b, b, b, b), fill=0)
        mask_crop = F.pad(mask_crop.unsqueeze(0), (b, b, b, b), fill=0).squeeze(0)

        return crop, mask_crop

def save_png_fast(img_chw_01, out_path):
    """
    img_chw_01: float tensor [3,H,W] on CUDA/CPU, values in [0,1].
    Uses compression_level=0 for speed.
    """
    img_u8 = (img_chw_01.clamp(0, 1) * 255.0).to(torch.uint8).cpu()
    write_png(img_u8, out_path, compression_level=0)

def save_mask_png_fast(mask_hw_u8, out_path):
    """
    mask_hw_u8: uint8 tensor [H,W] (0/1). Saves as single-channel PNG.
    """
    m = (mask_hw_u8 * 255).to(torch.uint8).unsqueeze(0).cpu()  # [1,H,W]
    write_png(m, out_path, compression_level=0)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Render prototype LEGO parts to images and masks (batched, CUDA-ready).")
    parser.add_argument('--part-ids', type=str,
                        help='Comma-separated list of LDraw IDs to render (e.g. "3001,3020,3626").')
    parser.add_argument('--views_per_obj', type=int, default=20,
                        help='How many views to render per part (default=20).')
    parser.add_argument('--size', type=int, default=330,
                        help='Output image size in pixels (square, default=330).')
    parser.add_argument('--render_scale', type=int, default=4,
                        help='Supersampling scale factor for rendering (default=4).')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device to use ("cpu" or "cuda", default=cpu).')
    parser.add_argument('--out_dir', type=str, default='viz_outputs/proto',
                        help='Directory to save rendered images and masks.')
    parser.add_argument('--batch', type=int, default=64,
                        help='Batch size (number of views rendered at once per mesh).')
    args = parser.parse_args()

    selected_ids = args.part_ids.split(',') if args.part_ids else PART_IDS

    # Initialize dataset/renderer
    ds = SyntheticBrickProtoDataset(
        obj_dir='objs',
        part_ids=selected_ids,
        image_size=args.size,
        render_scale=args.render_scale,
        views_per_obj=args.views_per_obj,
        device=args.device
    )
    os.makedirs(args.out_dir, exist_ok=True)

    # Main batched rendering loop: per mesh, render views_per_obj in chunks of args.batch
    torch.cuda.synchronize() if args.device.startswith('cuda') else None

    for mesh_idx, pid in enumerate(selected_ids):
        remaining = args.views_per_obj
        next_view_num = 1
        while remaining > 0:
            N = min(args.batch, remaining)
            # Sample a batch of camera angles on device
            azim, elev = ds.sample_azim_elev(N)
            # Render batch for this mesh
            imgs_bhwc, masks_bhw = ds.render_batch_views(mesh_idx, azim, elev)  # [N,H,W,3], [N,H,W]

            # Post-process and save per sample
            for i in range(N):
                # [H,W,3] -> [3,H,W]
                img_chw = imgs_bhwc[i].permute(2, 0, 1)
                mask_hw = masks_bhw[i]

                # Composite is already done by renderer background=black; mask keeps consistency
                # Tight crop -> square -> border (all on device)
                img_crop, mask_crop = ds._crop_square_with_border(img_chw, mask_hw, border_px=45)

                # Resize to output size (on device)
                img_out = F.resize(img_crop, [ds.image_size, ds.image_size])
                mask_out = F.resize(mask_crop.unsqueeze(0).float(),
                                    [ds.image_size, ds.image_size],
                                    interpolation=F.InterpolationMode.NEAREST)[0].to(torch.uint8)

                # Save (will move to CPU just-in-time)
                view_idx = next_view_num + i
                img_name = f"{pid}_{view_idx:02d}.png"
                mask_name = f"{pid}_{view_idx:02d}_mask.png"
                save_png_fast(img_out, os.path.join(args.out_dir, img_name))
                save_mask_png_fast(mask_out, os.path.join(args.out_dir, mask_name))

            next_view_num += N
            remaining -= N

    torch.cuda.synchronize() if args.device.startswith('cuda') else None
    print(f"Rendered {len(selected_ids) * args.views_per_obj} samples to {args.out_dir}")
