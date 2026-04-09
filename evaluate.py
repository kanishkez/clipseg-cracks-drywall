"""
Evaluate predictions: compute mIoU and Dice per-prompt and overall.

Generates tables, visualizations, and saves results.

Usage:
    python evaluate.py [--predictions predictions/] [--split test]
"""
import os
import argparse
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DATASET_PROMPTS = {
    "cracks": ["segment crack", "segment wall crack"],
    "drywall": ["segment taping area", "segment joint/tape", "segment drywall seam"],
}


def compute_iou(pred, gt, smooth=1e-6):
    """Compute IoU between two binary masks."""
    pred_bin = pred > 127
    gt_bin = gt > 127
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return (intersection + smooth) / (union + smooth)


def compute_dice(pred, gt, smooth=1e-6):
    """Compute Dice coefficient between two binary masks."""
    pred_bin = pred > 127
    gt_bin = gt > 127
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    total = pred_bin.sum() + gt_bin.sum()
    return (2.0 * intersection + smooth) / (total + smooth)


def get_gt_mask_path(dataset_name, image_filename, split="test"):
    """Get ground truth mask path for an image."""
    mask_path = os.path.join(
        DATA_DIR, dataset_name, split, "masks", image_filename + "_mask.png"
    )
    return mask_path if os.path.exists(mask_path) else None


def get_image_path(dataset_name, image_filename, split="test"):
    """Get original image path."""
    img_dir = os.path.join(DATA_DIR, dataset_name, split)
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        path = os.path.join(img_dir, image_filename + ext)
        if os.path.exists(path):
            return path
    return None


def evaluate_dataset(dataset_name, split="test"):
    """Evaluate all predictions for a dataset."""
    pred_dir = os.path.join(PREDICTIONS_DIR, dataset_name)
    if not os.path.exists(pred_dir):
        print(f"  [SKIP] No predictions for {dataset_name}")
        return {}

    results = {}
    prompts = DATASET_PROMPTS.get(dataset_name, [])

    for prompt in prompts:
        prompt_clean = prompt.replace(" ", "_").replace("/", "-")
        iou_scores = []
        dice_scores = []

        # Find all prediction files for this prompt
        pred_files = [
            f for f in os.listdir(pred_dir)
            if f.endswith(f"__{prompt_clean}.png")
        ]

        if not pred_files:
            print(f"    [SKIP] No predictions for prompt '{prompt}'")
            continue

        for pred_file in tqdm(pred_files, desc=f"    {prompt}", leave=False):
            # Extract image filename from prediction filename
            image_base = pred_file.replace(f"__{prompt_clean}.png", "")

            # Find ground truth mask
            gt_path = get_gt_mask_path(dataset_name, image_base, split)
            if gt_path is None:
                continue

            # Load masks
            pred_mask = np.array(Image.open(os.path.join(pred_dir, pred_file)).convert("L"))
            gt_mask = np.array(Image.open(gt_path).convert("L"))

            # Resize pred to match GT if needed
            if pred_mask.shape != gt_mask.shape:
                pred_mask = np.array(
                    Image.fromarray(pred_mask).resize(
                        (gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST
                    )
                )

            iou = compute_iou(pred_mask, gt_mask)
            dice = compute_dice(pred_mask, gt_mask)
            iou_scores.append(iou)
            dice_scores.append(dice)

        if iou_scores:
            results[prompt] = {
                "mIoU": float(np.mean(iou_scores)),
                "std_IoU": float(np.std(iou_scores)),
                "mDice": float(np.mean(dice_scores)),
                "std_Dice": float(np.std(dice_scores)),
                "n_samples": len(iou_scores),
            }
            print(
                f"    '{prompt}': mIoU={results[prompt]['mIoU']:.4f} "
                f"(±{results[prompt]['std_IoU']:.4f}), "
                f"mDice={results[prompt]['mDice']:.4f} "
                f"(±{results[prompt]['std_Dice']:.4f}), "
                f"n={len(iou_scores)}"
            )

    return results


def create_visualizations(dataset_name, split="test", n_samples=5):
    """Create side-by-side visualization of predictions."""
    pred_dir = os.path.join(PREDICTIONS_DIR, dataset_name)
    vis_dir = os.path.join(RESULTS_DIR, "visualizations", dataset_name)
    os.makedirs(vis_dir, exist_ok=True)

    prompts = DATASET_PROMPTS.get(dataset_name, [])
    if not prompts or not os.path.exists(pred_dir):
        return

    prompt = prompts[0]  # Use first prompt for visualization
    prompt_clean = prompt.replace(" ", "_").replace("/", "-")

    pred_files = [
        f for f in sorted(os.listdir(pred_dir))
        if f.endswith(f"__{prompt_clean}.png")
    ][:n_samples]

    for pred_file in pred_files:
        image_base = pred_file.replace(f"__{prompt_clean}.png", "")

        # Find original image and GT mask
        img_path = get_image_path(dataset_name, image_base, split)
        gt_path = get_gt_mask_path(dataset_name, image_base, split)

        if img_path is None:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        img = Image.open(img_path).convert("RGB")
        axes[0].imshow(img)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        # Ground truth
        if gt_path:
            gt = np.array(Image.open(gt_path).convert("L"))
            axes[1].imshow(gt, cmap="gray", vmin=0, vmax=255)
            axes[1].set_title("Ground Truth")
        else:
            axes[1].text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=20)
            axes[1].set_title("Ground Truth (N/A)")
        axes[1].axis("off")

        # Prediction
        pred = np.array(Image.open(os.path.join(pred_dir, pred_file)).convert("L"))
        axes[2].imshow(pred, cmap="gray", vmin=0, vmax=255)
        axes[2].set_title(f"Prediction: '{prompt}'")
        axes[2].axis("off")

        plt.suptitle(f"{dataset_name} - {image_base}", fontsize=14)
        plt.tight_layout()

        save_path = os.path.join(vis_dir, f"vis_{image_base}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"    Saved {len(pred_files)} visualizations to {vis_dir}")


def print_results_table(all_results):
    """Print formatted results table."""
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"{'Dataset':<12} {'Prompt':<28} {'mIoU':>8} {'±std':>8} {'mDice':>8} {'±std':>8} {'N':>5}")
    print(f"{'-'*80}")

    all_iou = []
    all_dice = []

    for dataset_name, results in all_results.items():
        for prompt, metrics in results.items():
            print(
                f"{dataset_name:<12} {prompt:<28} "
                f"{metrics['mIoU']:>8.4f} {metrics['std_IoU']:>8.4f} "
                f"{metrics['mDice']:>8.4f} {metrics['std_Dice']:>8.4f} "
                f"{metrics['n_samples']:>5d}"
            )
            all_iou.append(metrics["mIoU"])
            all_dice.append(metrics["mDice"])

    if all_iou:
        print(f"{'-'*80}")
        print(
            f"{'OVERALL':<12} {'(mean across prompts)':<28} "
            f"{np.mean(all_iou):>8.4f} {'':>8} "
            f"{np.mean(all_dice):>8.4f} {'':>8} "
            f"{'':>5}"
        )
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions")
    parser.add_argument(
        "--predictions",
        type=str,
        default=PREDICTIONS_DIR,
        help="Path to predictions directory",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--n_vis", type=int, default=5, help="Number of visualization samples"
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print("Evaluating Predictions")
    print(f"{'='*60}")

    all_results = {}

    for dataset_name in DATASET_PROMPTS:
        print(f"\n  Dataset: {dataset_name}")
        results = evaluate_dataset(dataset_name, args.split)
        if results:
            all_results[dataset_name] = results

    # Print summary table
    print_results_table(all_results)

    # Save results as JSON
    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Create visualizations
    print("\nGenerating visualizations...")
    for dataset_name in DATASET_PROMPTS:
        create_visualizations(dataset_name, args.split, n_samples=args.n_vis)

    # Save results as CSV for report
    csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    with open(csv_path, "w") as f:
        f.write("Dataset,Prompt,mIoU,std_IoU,mDice,std_Dice,N\n")
        for dataset_name, results in all_results.items():
            for prompt, metrics in results.items():
                f.write(
                    f"{dataset_name},{prompt},"
                    f"{metrics['mIoU']:.4f},{metrics['std_IoU']:.4f},"
                    f"{metrics['mDice']:.4f},{metrics['std_Dice']:.4f},"
                    f"{metrics['n_samples']}\n"
                )
    print(f"CSV results saved to {csv_path}")


if __name__ == "__main__":
    main()
