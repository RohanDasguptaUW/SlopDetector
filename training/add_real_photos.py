"""Create a standalone HuggingFace dataset from training/real_photos/*.jpg.

All photos are labelled Label_A=0 (real).
Images are processed in batches of 100 to limit peak memory use.
Output saved to datasets/real_photos_only (relative to repo root).

Usage:
    python training/add_real_photos.py
"""

import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value, concatenate_datasets
from PIL import Image as PILImage

PHOTOS_DIR = Path(__file__).parent / "real_photos"
DATASET_OUT = Path(__file__).parents[1] / "datasets" / "real_photos_only"
BATCH_SIZE = 100

FEATURES = Features({"Image": HFImage(), "Label_A": Value("int64")})


def _make_batch_dataset(paths: list[Path]) -> Dataset:
    rows: dict[str, list] = {"Image": [], "Label_A": []}
    for path in paths:
        rows["Image"].append(PILImage.open(path).convert("RGB"))
        rows["Label_A"].append(0)
    return Dataset.from_dict(rows, features=FEATURES)


def main() -> None:
    jpg_paths = sorted(PHOTOS_DIR.glob("*.jpg"))
    if not jpg_paths:
        sys.exit(f"Error: no .jpg files found in {PHOTOS_DIR}")

    total = len(jpg_paths)
    num_batches = -(-total // BATCH_SIZE)
    print(f"Found {total} photos — processing in {num_batches} batches of {BATCH_SIZE}")

    batch_datasets: list[Dataset] = []
    for i in range(0, total, BATCH_SIZE):
        batch = jpg_paths[i:i + BATCH_SIZE]
        batch_datasets.append(_make_batch_dataset(batch))
        done = min(i + BATCH_SIZE, total)
        print(f"  batch {len(batch_datasets)}/{num_batches}: loaded {done}/{total}")

    ds = concatenate_datasets(batch_datasets)
    print(f"\nTotal rows: {len(ds):,}  features={ds.features}")

    print(f"Saving to {DATASET_OUT} …")
    DATASET_OUT.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(DATASET_OUT))
    print("Done.")


if __name__ == "__main__":
    main()
