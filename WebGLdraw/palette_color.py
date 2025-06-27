import csv

color_rgb = {}
with open("colors.csv") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames

    # see if we have hex or separate channels
    if "rgb" in fields:
        # hex is like "#AABBCC" or "AABBCC"
        for row in reader:
            cid = int(row["id"])
            h = row["rgb"].lstrip("#")
            r = int(h[0:2], 16) / 255.0
            g = int(h[2:4], 16) / 255.0
            b = int(h[4:6], 16) / 255.0
            color_rgb[cid] = (r, g, b)
    elif all(c in fields for c in ("r","g","b")):
        for row in reader:
            cid = int(row["id"])
            r = int(row["r"]) / 255.0
            g = int(row["g"]) / 255.0
            b = int(row["b"]) / 255.0
            color_rgb[cid] = (r, g, b)
    else:
        raise RuntimeError(f"Unexpected columns in colors.csv: {fields}")

print(f"Loaded {len(color_rgb)} colors")
