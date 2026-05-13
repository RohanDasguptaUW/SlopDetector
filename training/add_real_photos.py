"""Augment the MS COCOAI dataset with real photos labelled Label_A=0.

Loads  : ~/datasets/ms_cocoai       (HuggingFace DatasetDict)
Adds   : training/real_photos/*.jpg  as Label_A=0 to the train split
Saves  : ~/datasets/ms_cocoai_augmented

Any column in the original dataset that isn't Image or Label_A is filled
with None for the new rows so that the schema stays consistent.

Usage:
    python training/add_real_photos.py
"""

import sys
from pathlib import Path

from datasets import (
    DatasetDict,
    Dataset,
    Image as HFImage,
    Value,
    concatenate_datasets,
    load_from_disk,
)
from PIL import Image as PILImage

DATASET_IN = Path(__file__).parents[1] / "datasets" / "ms_cocoai"
DATASET_OUT = Path(__file__).parents[1] / "datasets" / "ms_cocoai_augmented"
PHOTOS_DIR = Path(__file__).parent / "real_photos"
BATCH_SIZE = 100


def _make_batch_dataset(paths: list[Path], features) -> Dataset:
    rows: dict[str, list] = {col: [] for col in features}
    for path in paths:
        img = PILImage.open(path).convert("RGB")
        for col in features:
            if col == "Image":
                rows[col].append(img)
            elif col == "Label_A":
                rows[col].append(0)
            else:
                rows[col].append(None)
    return Dataset.from_dict(rows, features=features)


def load_real_photos(features) -> Dataset:
    """
    Build a Dataset from training/real_photos/*.jpg matching `features`.
    Images are processed in batches of BATCH_SIZE to limit peak memory use.
    Label_A → 0 (real). All other columns → None.
    """
    jpg_paths = sorted(PHOTOS_DIR.glob("*.jpg"))
    if not jpg_paths:
        sys.exit(f"Error: no .jpg files found in {PHOTOS_DIR}")

    total = len(jpg_paths)
    num_batches = -(-total // BATCH_SIZE)  # ceiling division
    print(f"Found {total} photos — processing in {num_batches} batches of {BATCH_SIZE}")

    batch_datasets: list[Dataset] = []
    for i in range(0, total, BATCH_SIZE):
        batch = jpg_paths[i:i + BATCH_SIZE]
        batch_ds = _make_batch_dataset(batch, features)
        batch_datasets.append(batch_ds)
        done = min(i + BATCH_SIZE, total)
        print(f"  batch {len(batch_datasets)}/{num_batches}: loaded {done}/{total}")

    return concatenate_datasets(batch_datasets)


def main() -> None:
    if not DATASET_IN.exists():
        sys.exit(f"Error: dataset not found at {DATASET_IN}")

    print(f"Loading dataset from {DATASET_IN} …")
    ds = load_from_disk(str(DATASET_IN))

    if not isinstance(ds, DatasetDict):
        sys.exit("Error: expected a DatasetDict with train/validation splits.")
    if "train" not in ds:
        sys.exit("Error: dataset has no 'train' split.")

    train = ds["train"]
    features = train.features
    print(f"train    : {len(train):,} rows  features={features}")
    if "validation" in ds:
        print(f"validation: {len(ds['validation']):,} rows")

    # Validate required columns exist
    for col in ("Image", "Label_A"):
        if col not in features:
            sys.exit(f"Error: expected column '{col}' not found in dataset. Got: {list(features)}")

    print()
    new_ds = load_real_photos(features)
    print(f"New rows : {len(new_ds):,}")

    print("Concatenating with train split …")
    augmented_train = concatenate_datasets([train, new_ds])
    print(f"Augmented train: {len(augmented_train):,} rows")

    # Reassemble DatasetDict — validation and any other splits are unchanged
    splits = {split: ds[split] for split in ds if split != "train"}
    splits["train"] = augmented_train
    augmented = DatasetDict(splits)

    print(f"\nSaving to {DATASET_OUT} …")
    augmented.save_to_disk(str(DATASET_OUT))

    print("\nDone.")
    for split, d in augmented.items():
        label_counts = d.to_pandas()["Label_A"].value_counts().to_dict()
        print(f"  {split}: {len(d):,} rows  label counts={label_counts}")


if __name__ == "__main__":
    main()
