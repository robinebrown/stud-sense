"""
render_dataset.py

This script renders two passes per view:
  1) A flat-color JPEG render without transparency (world + lighting)
  2) A grayscale PNG mask from the object-index pass

Usage example:
blender --background --python render_dataset.py -- \
    --input_dir ./objs \
    --output_rgb ./dataset/rgb \
    --output_mask ./dataset/mask \
    --views 100
"""

import bpy
import bmesh
import os, sys, argparse
from mathutils import Euler, Vector
import random

# --- Argument parsing ----------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',  required=True)
    parser.add_argument('--output_rgb',  required=True)
    parser.add_argument('--output_mask', required=True)
    parser.add_argument('--views',       type=int, default=100)
    return parser.parse_args(sys.argv[sys.argv.index("--")+1:])

# --- Manual OBJ importer with proper normals -----------------
def import_obj_manually(filepath, name):
    verts, faces = [], []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                _, x, y, z = line.split()[:4]
                verts.append((float(x), float(y), float(z)))
            elif line.startswith('f '):
                idxs = [int(part.split('//')[0]) - 1 for part in line.split()[1:]]
                faces.append(idxs)

    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    # Recalculate normals and smooth shading via bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for face in bm.faces:
        face.smooth = True
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj

# --- Scene setup ---------------------------------------------
def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # enable object-index pass
    scene.view_layers[0].use_pass_object_index = True
    # disable transparent film
    scene.render.film_transparent = False
    # resolution
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100

    # dark gray world background
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs[0].default_value = (0.2, 0.2, 0.2, 1)
        bg.inputs[1].default_value = 1.0

    # add camera (will position per object)
    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    # add key light
    key_light = bpy.data.lights.new("Key", type='SUN')
    key_light.energy = 5.0
    key_obj = bpy.data.objects.new("Key", key_light)
    scene.collection.objects.link(key_obj)
    key_obj.rotation_euler = Euler((1.2, 0.0, -0.8), 'XYZ')

    # add fill light
    fill_light = bpy.data.lights.new("Fill", type='AREA')
    fill_light.energy = 3.0
    fill_light.size = 5.0
    fill_obj = bpy.data.objects.new("Fill", fill_light)
    scene.collection.objects.link(fill_obj)
    fill_obj.rotation_euler = Euler((0.7, 0.7, 0), 'XYZ')

# --- Import & Render -----------------------------------------
def import_and_render(obj_path, rgb_out, mask_out, views):
    scene = bpy.context.scene
    name  = os.path.splitext(os.path.basename(obj_path))[0]

    # prepare Cycles engine
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64

    # pick one flat color per piece
    base_color = (random.random(), random.random(), random.random(), 1)

    for i in range(views):
        # import and center object
        obj = import_obj_manually(obj_path, name)
        obj.location = (0, 0, 0)
        obj.pass_index = 1

        # apply Principled material
        mat = bpy.data.materials.new(name + "_mat")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.inputs['Base Color'].default_value = base_color
        bsdf.inputs['Roughness'].default_value = 0.4
        out_node = nodes.new(type='ShaderNodeOutputMaterial')
        links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
        obj.data.materials.append(mat)

        # random rotation
        obj.rotation_euler = Euler((
            random.uniform(0, 3.14),
            random.uniform(0, 3.14),
            random.uniform(0, 3.14)
        ), 'XYZ')

        # fit camera orthographically
        dims = obj.dimensions
        max_dim = max(dims.x, dims.y, dims.z)
        cam = scene.camera
        cam.data.type = 'ORTHO'
        cam.data.ortho_scale = max_dim * 2.5
        cam.location = Vector((0, -max_dim * 2.0, max_dim))
        cam.rotation_euler = Euler((1.1, 0, 0), 'XYZ')

        # PASS 1: direct color render (JPEG)
        scene.use_nodes = False
        scene.render.image_settings.file_format = 'JPEG'
        scene.render.image_settings.quality = 80
        scene.render.filepath = os.path.join(rgb_out, f"{name}_{i:04d}.jpg")
        bpy.ops.render.render(write_still=True)

        # PASS 2: mask via compositor (PNG)
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()
        rl = tree.nodes.new('CompositorNodeRLayers')
        rl.location = (-200, 200)
        vout = tree.nodes.new('CompositorNodeOutputFile')
        vout.base_path = mask_out
        vout.file_slots[0].path = f"{name}_{i:04d}"
        vout.format.file_format = 'PNG'
        vout.format.color_mode  = 'BW'
        vout.format.compression  = 90
        vout.location = (200, 200)
        idx_out = next((o for o in rl.outputs if 'Index' in o.name), None)
        if idx_out is None:
            raise RuntimeError("No object-index output on RLayers")
        tree.links.new(idx_out, vout.inputs[0])
        bpy.ops.render.render(write_still=True)

        # cleanup
        bpy.data.meshes.remove(obj.data, do_unlink=True)
        bpy.data.materials.remove(mat, do_unlink=True)

# --- Main ----------------------------------------------------
def main():
    args = parse_args()
    args.output_rgb = os.path.abspath(args.output_rgb)
    args.output_mask = os.path.abspath(args.output_mask)
    os.makedirs(args.output_rgb, exist_ok=True)
    os.makedirs(args.output_mask, exist_ok=True)

    setup_scene()
    for fname in os.listdir(args.input_dir):
        if not fname.lower().endswith('.obj'):
            continue
        import_and_render(
            os.path.join(args.input_dir, fname),
            args.output_rgb, args.output_mask, args.views)

if __name__ == '__main__':
    main()
