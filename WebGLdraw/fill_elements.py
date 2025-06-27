#!/usr/bin/env python3
import os
import re
import csv
from collections import defaultdict

def normalize_base(pid):
    """Drop leading zeros and anything after the first non-digit."""
    m = re.match(r'^0*([0-9]+)', pid)
    return m.group(1) if m else pid

# 1) Load palette & transparency flags from colors.csv
color_trans = {}
with open("colors.csv") as f:
    reader = csv.DictReader(f)
    has_flag = "is_trans" in reader.fieldnames
    for row in reader:
        cid = int(row["id"])
        if has_flag:
            color_trans[cid] = row["is_trans"] in ("1","True","true")
        else:
            # fallback: any color name containing “Trans”
            color_trans[cid] = ("Trans" in row.get("name",""))

# 2) Gather all mesh variants from objs/
mesh_variants = [
    fn[:-4] for fn in os.listdir("objs")
    if fn.lower().endswith(".obj")
]
base_to_variants = defaultdict(list)
for var in mesh_variants:
    base = normalize_base(var)
    base_to_variants[base].append(var)

# 3) Read inventory_parts.csv, filter to your bases and bucket colors
base_buckets = defaultdict(lambda: {"opaque": set(), "trans": set()})
with open("inventory_parts.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        base = normalize_base(row["part_num"])
        if base not in base_to_variants:
            continue
        cid = int(row["color_id"])
        bucket = "trans" if color_trans.get(cid, False) else "opaque"
        base_buckets[base][bucket].add(cid)

# 4) Write out elements.csv with one entry per variant + color
out_file = "elements.csv"
with open(out_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["part_num", "color_id", "is_trans"])
    for base, buckets in sorted(base_buckets.items()):
        for var in sorted(base_to_variants[base]):
            for cid in sorted(buckets["opaque"]):
                writer.writerow([var, cid, 0])
            for cid in sorted(buckets["trans"]):
                writer.writerow([var, cid, 1])

total_variants = sum(len(v) for v in base_to_variants.values())
print(f"✅ Wrote {out_file}: covered {total_variants} mesh variants")
