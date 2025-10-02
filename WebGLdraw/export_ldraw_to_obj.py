# export_ldraw_to_obj.py
import bpy, sys, os

# Blender passes args before “--” in sys.argv, everything after is ours
argv = sys.argv
argv = argv[argv.index("--") + 1:]

input_dir, output_dir = argv

# Ensure output folder exists
os.makedirs(output_dir, exist_ok=True)

# Recursively find .dat files
dat_paths = []
for root, _, files in os.walk(input_dir):
    for f in files:
        if f.lower().endswith(".dat"):
            dat_paths.append(os.path.join(root, f))

# Clear default scene
bpy.ops.wm.read_homefile(use_empty=True)

for dat_path in dat_paths:
    part_name = os.path.splitext(os.path.basename(dat_path))[0]
    out_path  = os.path.join(output_dir, part_name + ".obj")
    print(f"Importing {dat_path} → exporting {out_path}")

    # Import the LDraw .dat (requires the LDraw importer add-on)
    bpy.ops.import_scene.ldraw(filepath=dat_path)

    # Select all imported meshes
    for obj in bpy.context.scene.objects:
        obj.select_set(obj.type == 'MESH')

    # Export selected meshes to OBJ
    bpy.ops.export_scene.obj(
        filepath=out_path,
        use_selection=True,
        axis_forward='Y',
        axis_up='Z',
        use_materials=False
    )

    # Remove all objects before next iteration
    bpy.ops.object.delete()

print("Export complete.")
