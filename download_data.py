"""
Download datasets from Roboflow and convert annotations to binary masks.

Handles both COCO segmentation (polygon) and COCO object-detection (bbox) formats.
When only bounding boxes are available, creates filled bbox masks.

Usage:
    export ROBOFLOW_API_KEY="your_api_key"
    python3 download_data.py
"""
import os
import json
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DATASETS = {
    "cracks": {
        "workspace": "fyp-ny1jt",
        "project": "cracks-3ii36",
        "version": None,  # Will auto-detect or generate
        "prompt_class": "crack",
    },
    "drywall": {
        "workspace": "objectdetect-pu6rn",
        "project": "drywall-join-detect",
        "version": 2,
        "prompt_class": "taping",
    },
}


def download_datasets():
    """Download both datasets from Roboflow."""
    from roboflow import Roboflow

    if not ROBOFLOW_API_KEY:
        raise ValueError(
            "Please set ROBOFLOW_API_KEY environment variable.\n"
            "  export ROBOFLOW_API_KEY='your_key_here'"
        )

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    os.makedirs(DATA_DIR, exist_ok=True)

    for name, cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Downloading dataset: {name}")
        print(f"{'='*60}")

        project = rf.workspace(cfg["workspace"]).project(cfg["project"])
        print(f"  Project type: {project.type}")
        print(f"  Classes: {project.classes}")

        # Determine version
        version_num = cfg["version"]
        if version_num is None:
            # Check for existing versions
            version_info = project.get_version_information()
            if version_info:
                # Use the latest version
                version_id = version_info[0]["id"]
                version_num = int(version_id.split("/")[-1])
                print(f"  Using existing version: {version_num}")
            else:
                # Generate a new version with standard preprocessing
                print("  No versions found. Generating version 1...")
                project.generate_version(
                    settings={
                        "preprocessing": {
                            "auto-orient": True,
                            "resize": {"width": 640, "height": 640, "format": "Stretch to"},
                        },
                        "augmentation": {},
                    }
                )
                version_num = 1
                print(f"  Generated version {version_num}")

        version = project.version(version_num)

        # Try coco-segmentation first, fall back to coco (which gives bboxes)
        download_format = "coco"
        dest = os.path.join(DATA_DIR, name)
        
        try:
            dataset = version.download(download_format, location=dest)
            print(f"  -> Downloaded ({download_format}) to: {dataset.location}")
        except Exception as e:
            print(f"  Error with {download_format}: {e}")
            # Fallback
            download_format = "coco"
            dataset = version.download(download_format, location=dest)
            print(f"  -> Downloaded ({download_format}) to: {dataset.location}")


def coco_to_binary_masks(dataset_name, split):
    """
    Convert COCO annotations to binary mask PNGs.
    Supports both polygon segmentation and bounding box annotations.
    """
    base_dir = os.path.join(DATA_DIR, dataset_name, split)
    ann_file = os.path.join(base_dir, "_annotations.coco.json")

    if not os.path.exists(ann_file):
        print(f"  [SKIP] No annotation file at {ann_file}")
        return 0

    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    mask_dir = os.path.join(base_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    # Build image id -> info mapping
    img_map = {img["id"]: img for img in coco_data["images"]}

    # Group annotations by image
    img_anns = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    count = 0
    has_segmentation = False

    # Check if any annotation has polygon segmentation
    for ann in coco_data["annotations"][:10]:
        if "segmentation" in ann and ann["segmentation"]:
            if isinstance(ann["segmentation"], list) and len(ann["segmentation"]) > 0:
                has_segmentation = True
                break

    method = "polygon" if has_segmentation else "bbox"
    print(f"  Mask creation method: {method}")

    for img_id, img_info in tqdm(img_map.items(), desc=f"  {dataset_name}/{split}", leave=False):
        h, w = img_info["height"], img_info["width"]
        anns = img_anns.get(img_id, [])

        # Create combined binary mask
        mask_img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask_img)

        for ann in anns:
            if has_segmentation and "segmentation" in ann and ann["segmentation"]:
                # Polygon segmentation
                seg = ann["segmentation"]
                if isinstance(seg, list):
                    for poly in seg:
                        if len(poly) >= 6:  # At least 3 points
                            # Convert flat list to list of tuples
                            points = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
                            draw.polygon(points, fill=255)
            elif "bbox" in ann:
                # Bounding box: [x, y, width, height]
                x, y, bw, bh = ann["bbox"]
                draw.rectangle([x, y, x + bw, y + bh], fill=255)

        # Save mask
        mask_array = np.array(mask_img)
        mask_binary = ((mask_array > 0).astype(np.uint8)) * 255
        mask_filename = os.path.splitext(img_info["file_name"])[0] + "_mask.png"
        Image.fromarray(mask_binary, mode="L").save(
            os.path.join(mask_dir, mask_filename)
        )
        count += 1

    return count


def convert_all_masks():
    """Convert annotations for all datasets and splits."""
    print(f"\n{'='*60}")
    print("Converting COCO annotations to binary masks")
    print(f"{'='*60}")

    for dataset_name in DATASETS:
        for split in ["train", "valid", "test"]:
            n = coco_to_binary_masks(dataset_name, split)
            if n > 0:
                print(f"  {dataset_name}/{split}: {n} masks created")


def verify_data():
    """Verify downloaded data integrity."""
    print(f"\n{'='*60}")
    print("Verifying data integrity")
    print(f"{'='*60}")

    for dataset_name in DATASETS:
        for split in ["train", "valid", "test"]:
            img_dir = os.path.join(DATA_DIR, dataset_name, split)
            mask_dir = os.path.join(img_dir, "masks")

            if not os.path.exists(img_dir):
                print(f"  [WARN] {dataset_name}/{split}: directory missing")
                continue

            imgs = [
                f
                for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
                and not f.endswith("_mask.png")
                and f != "_annotations.coco.json"
            ]
            masks = (
                [f for f in os.listdir(mask_dir) if f.endswith("_mask.png")]
                if os.path.exists(mask_dir)
                else []
            )

            print(f"  {dataset_name}/{split}: {len(imgs)} images, {len(masks)} masks")

            # Verify mask values
            if masks:
                sample_mask = np.array(
                    Image.open(os.path.join(mask_dir, masks[0]))
                )
                unique_vals = np.unique(sample_mask)
                assert all(
                    v in [0, 255] for v in unique_vals
                ), f"Invalid mask values: {unique_vals}"
                print(f"    Mask values OK: {unique_vals}")


if __name__ == "__main__":
    download_datasets()
    convert_all_masks()
    verify_data()
    print("\n✅ Data pipeline complete!")
