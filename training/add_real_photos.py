"""Create a CSV manifest of real photos for training.

Writes datasets/real_photos_manifest.csv with columns:
    filepath  — absolute path to each JPEG
    Label_A   — 0 (real) for all rows

No images are loaded into memory.

Usage:
    python training/add_real_photos.py
"""

import csv
import sys
from pathlib import Path

PHOTOS_DIR = Path(__file__).parent / "real_photos"
MANIFEST_OUT = Path(__file__).parents[1] / "datasets" / "real_photos_manifest.csv"


def main() -> None:
    jpg_paths = sorted(PHOTOS_DIR.glob("*.jpg"))
    if not jpg_paths:
        sys.exit(f"Error: no .jpg files found in {PHOTOS_DIR}")

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "Label_A"])
        for path in jpg_paths:
            writer.writerow([path.resolve(), 0])

    print(f"Wrote {len(jpg_paths):,} rows to {MANIFEST_OUT}")


if __name__ == "__main__":
    main()
