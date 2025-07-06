import os
import argparse
import random
import torch
import torchvision.utils as vutils
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


def main():
    parser = argparse.ArgumentParser(
        description="Render a single LEGO .obj part into multiple views with YOLO labels"
    )
    parser.add_argument(
        "--obj_path", type=str, required=True,
        help="Path to the single .obj file to render"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Directory to save rendered images, masks, and labels"
    )
    parser.add_argument(
        "--views", type=int, default=12,
        help="Number of random views to render"
    )
    parser.add_argument(
        "--size", type=int, default=330,
        help="Square image size (pixels)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Renderer device: cpu, cuda, or mps"
    )
    args = parser.parse_args()

    # Prepare output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Derive part ID from filename
    pid = os.path.splitext(os.path.basename(args.obj_path))[0]

    # Load mesh
    device = torch.device(args.device)
    mesh = load_objs_as_meshes([args.obj_path], device=device)

    # Apply constant gray texture
    gray_val = 0.8
    verts_features = [torch.full((verts.shape[0], 3), gray_val, device=device)
                      for verts in mesh.verts_list()]
    mesh.textures = TexturesVertex(verts_features=verts_features)

    # Set up renderer
    cameras = FoVPerspectiveCameras(device=device)
    raster_settings = RasterizationSettings(
        image_size=args.size,
        blur_radius=0.0,
        faces_per_pixel=1
    )
    lights = DirectionalLights(device=device)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )

    # Precompute random camera poses
    view_poses = [look_at_view_transform(
        dist=2.5,
        elev=random.uniform(10, 60),
        azim=random.uniform(0, 360)
    ) for _ in range(args.views)]

    # Render loop
    for idx, (R, T) in enumerate(view_poses, start=1):
        # Render color image and mask
        images = renderer(
            mesh.extend(1),
            cameras=FoVPerspectiveCameras(device=device, R=R, T=T)
        )  # (1, H, W, 3)
        img = images[0, ..., :3].permute(2, 0, 1)  # C×H×W
        mask = (images[0, ..., :3].sum(-1) > 0).to(torch.uint8)  # H×W

        # Save outputs
        img_path = os.path.join(args.out_dir, f"{pid}_{idx:02d}.png")
        mask_path = os.path.join(args.out_dir, f"{pid}_{idx:02d}_mask.png")
        label_path = os.path.join(args.out_dir, f"{pid}_{idx:02d}.txt")

        vutils.save_image(img, img_path)
        vutils.save_image(mask.unsqueeze(0).float(), mask_path)

        # Compute tight YOLO bbox
        ys, xs = torch.nonzero(mask, as_tuple=True)
        if ys.numel() == 0:
            # Skip if nothing rendered
            continue
        y0, y1 = ys.min().item(), ys.max().item()
        x0, x1 = xs.min().item(), xs.max().item()
        H, W = mask.shape
        x_center = ((x0 + x1) / 2) / W
        y_center = ((y0 + y1) / 2) / H
        w_norm = (x1 - x0) / W
        h_norm = (y1 - y0) / H

        with open(label_path, "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    print(f"Rendered {idx} views for part {pid} into {args.out_dir}")


if __name__ == "__main__":
    main()
