import os, csv

# 1) Gather all meshes
mesh_parts = {fn[:-4] for fn in os.listdir("objs") if fn.endswith(".obj")}

# 2) Load the elements mappings
part_to_cids = {}
with open("elements.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pid = row["part_num"]
        cid = int(row["color_id"])
        part_to_cids.setdefault(pid, set()).add(cid)

# 3) Stats
mapped_parts = set(part_to_cids.keys())
missing_parts = sorted(mesh_parts - mapped_parts)
extra_mapped = sorted(mapped_parts - mesh_parts)

print(f"{len(mapped_parts)} parts with at least one real color mapping")
print(f"{len(missing_parts)} parts with NO mapping  →  e.g. {missing_parts[:10]} …")
print(f"{len(extra_mapped)} mapped parts not in objs/ → e.g. {extra_mapped[:10]} …")

with open("unmapped_parts.txt","w") as f:
    for pid in missing_parts:
        f.write(pid + "\n")
