# dump_unmapped.py
import os, csv, re
from collections import defaultdict

def normalize(pid): 
    m = re.match(r'^0*([0-9]+)', pid)
    return m.group(1) if m else pid

# load meshes
meshes = {fn[:-4] for fn in os.listdir("objs") if fn.endswith(".obj")}
# load mapped parts
mapped = set()
with open("elements.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        mapped.add(r["part_num"])
# find unmapped
unmapped = sorted(meshes - mapped)
with open("unmapped_parts.txt","w") as out:
    out.write("\n".join(unmapped))
print(f"Wrote {len(unmapped)} unmapped part IDs to unmapped_parts.txt")
