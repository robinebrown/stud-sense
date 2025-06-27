#!/usr/bin/env python3
import os
import pandas as pd

# === CSVs live alongside this script in the same folder ===
COLORS_CSV    = "colors.csv"      # uncompressed or .gz
ELEMENTS_CSV  = "elements.csv"    # uncompressed or .gz
# =========================================================

def load_csv(path):
    if path.endswith(".gz"):
        return pd.read_csv(path, compression="gzip")
    else:
        return pd.read_csv(path)

def main():
    # 1) Load the tables
    print(f"Loading colors from {COLORS_CSV}…")
    colors   = load_csv(COLORS_CSV)
    print(f"Loading elements from {ELEMENTS_CSV}…")
    elements = load_csv(ELEMENTS_CSV)

    # 2) Find all transparent color IDs
    trans_color_ids = set(colors.loc[colors['is_trans'], 'id'])
    print(f"Found {len(trans_color_ids)} transparent colors.")

    # 3) Filter elements that use those colors
    te = elements[elements['color_id'].isin(trans_color_ids)]
    part_nums = sorted(te['part_num'].unique())
    print(f"Found {len(part_nums)} unique parts with at least one transparent variant.")

    # 4) Write out the .obj list
    out_path = "parts_with_transparency.txt"
    with open(out_path, "w") as f:
        for pn in part_nums:
            f.write(f"{pn}.obj\n")
    print(f"Wrote parts list to {out_path}")

if __name__ == "__main__":
    main()
