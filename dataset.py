"""
Unified PyTorch Dataset for text-conditioned segmentation.

Produces (image, text_prompt, binary_mask) triplets from both
the Cracks and Drywall-Join-Detect datasets.
"""
import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, ConcatDataset

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Prompt pools for each dataset
PROMPT_POOLS = {
    "cracks": [
        "segment crack",
        "segment wall crack",
        "segment cracks",
        "segment the crack",
        "segment crack region",
    ],
    "drywall": [
        "segment taping area",
        "segment joint/tape",
        "segment drywall seam",
        "segment the taping area",
        "segment tape joint",
    ],
}

# Input size for CLIPSeg
CLIPSEG_INPUT_SIZE = 352


class SegmentationDataset(Dataset):
    """
    Single-dataset loader for text-conditioned segmentation.

    Args:
        dataset_name: 'cracks' or 'drywall'
        split: 'train', 'valid', or 'test'
        processor: CLIPSegProcessor instance
        augment: whether to apply data augmentation (only for train)
    """

    def __init__(self, dataset_name, split, processor, augment=False):
        self.dataset_name = dataset_name
        self.split = split
        self.processor = processor
        self.augment = augment and (split == "train")
        self.prompts = PROMPT_POOLS[dataset_name]

        img_dir = os.path.join(DATA_DIR, dataset_name, split)
        mask_dir = os.path.join(img_dir, "masks")

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            raise FileNotFoundError(
                f"Dataset not found at {img_dir}. Run download_data.py first."
            )

        # Build list of (image_path, mask_path) pairs
        self.samples = []
        mask_files = {f for f in os.listdir(mask_dir) if f.endswith("_mask.png")}

        for mask_file in sorted(mask_files):
            # Derive image filename from mask filename
            base_name = mask_file.replace("_mask.png", "")
            # Try common image extensions
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                candidate = os.path.join(img_dir, base_name + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path is not None:
                self.samples.append(
                    (img_path, os.path.join(mask_dir, mask_file))
                )

        print(
            f"  {dataset_name}/{split}: loaded {len(self.samples)} samples"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Store original size for prediction
        orig_size = image.size  # (W, H)

        # Random prompt from pool
        prompt = random.choice(self.prompts)

        # Simple augmentation: random horizontal flip
        if self.augment and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Resize mask to CLIPSeg output size
        mask_resized = mask.resize(
            (CLIPSEG_INPUT_SIZE, CLIPSEG_INPUT_SIZE), Image.NEAREST
        )
        mask_tensor = torch.from_numpy(
            (np.array(mask_resized) > 127).astype(np.float32)
        )

        # Process image+text with CLIPSeg processor
        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        # Squeeze batch dimension
        pixel_values = inputs["pixel_values"].squeeze(0)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mask": mask_tensor,
            "prompt": prompt,
            "image_path": img_path,
            "orig_size": orig_size,
        }


def get_datasets(processor, split="train"):
    """
    Get combined dataset from both cracks and drywall datasets.

    Returns:
        ConcatDataset combining both datasets
    """
    augment = split == "train"
    datasets = []

    for dataset_name in PROMPT_POOLS:
        try:
            ds = SegmentationDataset(
                dataset_name, split, processor, augment=augment
            )
            if len(ds) > 0:
                datasets.append(ds)
        except FileNotFoundError as e:
            print(f"  [WARN] {e}")

    if not datasets:
        raise RuntimeError(f"No datasets found for split '{split}'")

    combined = ConcatDataset(datasets)
    print(f"  Combined {split}: {len(combined)} total samples")
    return combined


def custom_collate_fn(batch):
    """Custom collate to handle variable-length input_ids."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    masks = torch.stack([b["mask"] for b in batch])

    # Pad input_ids and attention_mask to same length
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].shape[0]
        input_ids[i, :seq_len] = b["input_ids"]
        attention_mask[i, :seq_len] = b["attention_mask"]

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mask": masks,
        "prompt": [b["prompt"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
        "orig_size": [b["orig_size"] for b in batch],
    }
