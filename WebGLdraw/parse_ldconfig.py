# parse_ldconfig.py

import re, csv

ldraw_palette = {}   # code → (r, g, b, alpha)
with open("LDConfig_text.txt", "r", errors="ignore") as f:
    for line in f:
        if "!COLOUR" not in line:
            continue
        # split on whitespace
        toks = re.split(r'\s+', line.strip())
        try:
            # find CODE <n>, VALUE <#RRGGBB>, optional ALPHA <x>
            idx_code  = toks.index("CODE") + 1
            idx_value = toks.index("VALUE") + 1
            code      = int(toks[idx_code])
            hexcol    = toks[idx_value].lstrip("#")
            r = int(hexcol[0:2], 16)/255.0
            g = int(hexcol[2:4], 16)/255.0
            b = int(hexcol[4:6], 16)/255.0

            if "ALPHA" in toks:
                idx_alpha = toks.index("ALPHA") + 1
                alpha     = int(toks[idx_alpha]) / 255.0
            else:
                alpha = 1.0

            ldraw_palette[code] = (r, g, b, alpha)
        except (ValueError, IndexError):
            # malformed line – skip
            continue

print(f"Parsed {len(ldraw_palette)} LDraw colours")

# Optional: inspect by writing a CSV
with open("ldraw_palette.csv","w", newline="") as out:
    w = csv.writer(out)
    w.writerow(["code","r","g","b","alpha"])
    for code,(r,g,b,a) in sorted(ldraw_palette.items()):
        w.writerow([code, r, g, b, a])
print("→ wrote ldraw_palette.csv")
