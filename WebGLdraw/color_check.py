import os, csv

# load all part IDs from your mesh folder
meshes = {fn[:-4] for fn in os.listdir("/Users/robinbrown/repos/stud-sense/WebGLdraw/objs") if fn.endswith(".obj")}

# load all part IDs that have at least one real-world color in elements.csv
mapped = set()
with open("elements.csv") as f:
    for row in csv.DictReader(f):
        mapped.add(row["part_num"])

missing = sorted(meshes - mapped)
print(f"{len(missing)} parts with no color data:", missing[:10], "…")
