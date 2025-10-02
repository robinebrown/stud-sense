from pathlib import Path
import sys
import os

src = Path("common_parts.txt")

# Ensure the output directory exists
out_dir = Path("common_parts")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "item_numbers.txt"

item_numbers = []
expect_item_number = False

with src.open("r", encoding="utf-8") as f:
    for raw in f:
        line = raw.strip()
        if not line:
            continue

        if expect_item_number:
            if line.isdigit():
                item_numbers.append(line)
                expect_item_number = False
            continue

        if not line.isdigit():
            expect_item_number = True

# Save to the new directory
out_file.write_text("\n".join(item_numbers), encoding="utf-8")

print(f"Extracted {len(item_numbers)} item numbers into {out_file}")