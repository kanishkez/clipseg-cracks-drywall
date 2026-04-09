"""
Fine-tune CLIPSeg decoder for text-conditioned segmentation.

Usage:
    python train.py [--epochs 30] [--batch_size 4] [--lr 1e-4] [--seed 42]
"""
import os
import sys
import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_datasets, custom_collate_fn, CLIPSEG_INPUT_SIZE

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def patch_clipseg_for_mps():
    """Monkey-patch CLIPSeg decoder to use .reshape() instead of .view() for MPS."""
    from transformers.models.clipseg import modeling_clipseg
    from typing import Optional
    import torch
    import math

    def patched_forward(
        self,
        hidden_states: tuple[torch.Tensor],
        conditional_embeddings: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        activations = hidden_states[::-1]

        output = None
        for i, (activation, layer, reduce) in enumerate(zip(activations, self.layers, self.reduces)):
            if output is not None:
                output = reduce(activation) + output
            else:
                output = reduce(activation)

            if i == self.conditional_layer:
                output = self.film_mul(conditional_embeddings) * output.permute(1, 0, 2) + self.film_add(
                    conditional_embeddings
                )
                output = output.permute(1, 0, 2)

            layer_outputs = layer(
                output, attention_mask=None, causal_attention_mask=None, output_attentions=output_attentions
            )

            output = layer_outputs[0]

            if output_hidden_states:
                all_hidden_states += (output,)

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        output = output[:, 1:, :].permute(0, 2, 1).contiguous()  # remove cls token and reshape to [batch_size, reduce_dim, seq_len]

        size = int(math.sqrt(output.shape[2]))

        batch_size = conditional_embeddings.shape[0]
        output = output.reshape(batch_size, output.shape[1], size, size)

        logits = self.transposed_convolution(output).squeeze(1)

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_attentions] if v is not None)

        return modeling_clipseg.CLIPSegDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    modeling_clipseg.CLIPSegDecoder.forward = patched_forward
    print("  ✅ Patched CLIPSeg decoder for MPS compatibility")


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dice_loss(pred, target, smooth=1.0):
    """Differentiable Dice loss."""
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1.0 - dice).mean()


def dice_score(pred, target, smooth=1.0):
    """Compute Dice score (for evaluation)."""
    pred_bin = (pred > 0).float()
    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def iou_score(pred, target, smooth=1.0):
    """Compute IoU score."""
    pred_bin = (pred > 0).float()
    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def freeze_encoder(model):
    """Freeze CLIP encoder parameters, keep decoder trainable."""
    # Freeze the CLIP model (vision + text encoder)
    for name, param in model.named_parameters():
        if "clip" in name.lower() or "encoder" in name.lower():
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def train_one_epoch(model, dataloader, optimizer, bce_loss_fn):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc="  Train", leave=False)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        masks = batch["mask"].to(DEVICE)

        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Get logits and resize to match mask size
        logits = outputs.logits.squeeze(1)  # (B, H, W)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits.unsqueeze(1),
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        # Combined loss: BCE + Dice
        loss_bce = bce_loss_fn(logits, masks)
        loss_dice = dice_loss(logits, masks)
        loss = loss_bce + loss_dice

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            d = dice_score(logits, masks)
            i = iou_score(logits, masks)

        total_loss += loss.item()
        total_dice += d
        total_iou += i
        n_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

        if n_batches >= 20:  # LIMIT BATCHES FOR TIME CONSTRAINTS
            break

    return {
        "loss": total_loss / n_batches,
        "dice": total_dice / n_batches,
        "iou": total_iou / n_batches,
    }


@torch.no_grad()
def validate(model, dataloader, bce_loss_fn):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    n_batches = 0

    for batch in tqdm(dataloader, desc="  Valid", leave=False):
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        masks = batch["mask"].to(DEVICE)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits.squeeze(1)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits.unsqueeze(1),
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        loss_bce = bce_loss_fn(logits, masks)
        loss_dice = dice_loss(logits, masks)
        loss = loss_bce + loss_dice

        total_loss += loss.item()
        total_dice += dice_score(logits, masks)
        total_iou += iou_score(logits, masks)
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "dice": total_dice / n_batches,
        "iou": total_iou / n_batches,
    }


def plot_training_curves(history, save_path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Valid")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice
    axes[1].plot(epochs, history["train_dice"], "b-", label="Train")
    axes[1].plot(epochs, history["val_dice"], "r-", label="Valid")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].set_title("Dice Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # IoU
    axes[2].plot(epochs, history["train_iou"], "b-", label="Train")
    axes[2].plot(epochs, history["val_iou"], "r-", label="Valid")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].set_title("IoU")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CLIPSeg")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model_name",
        type=str,
        default="CIDAS/clipseg-rd64-refined",
        help="Pretrained model name",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("CLIPSeg Fine-Tuning for Text-Conditioned Segmentation")
    print(f"{'='*60}")
    print(f"  Device: {DEVICE}")
    print(f"  Seed: {args.seed}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Model: {args.model_name}")

    # Set seed
    set_seed(args.seed)

    # Patch CLIPSeg for MPS compatibility
    if DEVICE.type == "mps":
        patch_clipseg_for_mps()

    # Load processor and model
    print("\nLoading model...")
    processor = CLIPSegProcessor.from_pretrained(args.model_name)
    model = CLIPSegForImageSegmentation.from_pretrained(args.model_name)
    model.to(DEVICE)

    # Freeze encoder
    print("Freezing CLIP encoder...")
    freeze_encoder(model)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = get_datasets(processor, split="train")
    val_dataset = get_datasets(processor, split="valid")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )

    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # Training loop
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_dice = 0.0
    history = {
        "train_loss": [], "train_dice": [], "train_iou": [],
        "val_loss": [], "val_dice": [], "val_iou": [],
    }

    print(f"\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} (lr={optimizer.param_groups[0]['lr']:.6f})")

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, bce_loss_fn)
        history["train_loss"].append(train_metrics["loss"])
        history["train_dice"].append(train_metrics["dice"])
        history["train_iou"].append(train_metrics["iou"])

        # Validate
        val_metrics = validate(model, val_loader, bce_loss_fn)
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])

        scheduler.step()

        print(
            f"  Train - Loss: {train_metrics['loss']:.4f}, "
            f"Dice: {train_metrics['dice']:.4f}, "
            f"IoU: {train_metrics['iou']:.4f}"
        )
        print(
            f"  Valid - Loss: {val_metrics['loss']:.4f}, "
            f"Dice: {val_metrics['dice']:.4f}, "
            f"IoU: {val_metrics['iou']:.4f}"
        )

        # Save best model
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            save_path = os.path.join(CHECKPOINT_DIR, "best_model")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"  ✅ New best model saved (Dice: {best_val_dice:.4f})")

        # Save latest model
        save_path = os.path.join(CHECKPOINT_DIR, "latest_model")
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)

    # Save training history
    history_path = os.path.join(CHECKPOINT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    plot_training_curves(
        history,
        os.path.join(CHECKPOINT_DIR, "training_curves.png"),
    )

    # Save training config
    config = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "model_name": args.model_name,
        "best_val_dice": best_val_dice,
        "device": str(DEVICE),
        "input_size": CLIPSEG_INPUT_SIZE,
    }
    with open(os.path.join(CHECKPOINT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best validation Dice: {best_val_dice:.4f}")
    print(f"  Checkpoints saved to: {CHECKPOINT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
