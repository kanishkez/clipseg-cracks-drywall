"""
Run inference with the fine-tuned CLIPSeg model.

Generates binary prediction masks as single-channel PNG files
with values {0, 255}. Filenames: {image_id}__{prompt}.png

Usage:
    python predict.py [--checkpoint checkpoints/best_model] [--threshold 0.5]
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from tqdm import tqdm

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

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

        output = output[:, 1:, :].permute(0, 2, 1).contiguous()
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

if DEVICE.type == "mps":
    patch_clipseg_for_mps()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")

# Prompts to predict with for each dataset
DATASET_PROMPTS = {
    "cracks": ["segment crack", "segment wall crack"],
    "drywall": ["segment taping area", "segment joint/tape", "segment drywall seam"],
}


def load_model(checkpoint_path):
    """Load fine-tuned model and processor."""
    print(f"Loading model from {checkpoint_path}...")
    processor = CLIPSegProcessor.from_pretrained(checkpoint_path)
    model = CLIPSegForImageSegmentation.from_pretrained(checkpoint_path)
    model.to(DEVICE)
    model.eval()
    return processor, model


def get_test_images(dataset_name, split="test"):
    """Get list of test image paths."""
    img_dir = os.path.join(DATA_DIR, dataset_name, split)
    if not os.path.exists(img_dir):
        print(f"  [WARN] {img_dir} not found, trying 'valid' split")
        img_dir = os.path.join(DATA_DIR, dataset_name, "valid")

    if not os.path.exists(img_dir):
        return []

    images = []
    for f in sorted(os.listdir(img_dir)):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            if not f.endswith("_mask.png") and f != "_annotations.coco.json":
                images.append(os.path.join(img_dir, f))

    return images


@torch.no_grad()
def predict_single(model, processor, image, prompt, threshold=0.5):
    """
    Predict binary mask for a single image+prompt pair.

    Returns:
        numpy array of shape (H, W) with values {0, 255}
    """
    orig_size = image.size  # (W, H)

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits.squeeze()  # (h, w)

    # Resize to original image size
    logits_resized = F.interpolate(
        logits.unsqueeze(0).unsqueeze(0),
        size=(orig_size[1], orig_size[0]),  # (H, W)
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    # Threshold to binary mask
    pred_mask = (torch.sigmoid(logits_resized) > threshold).cpu().numpy()
    mask_uint8 = (pred_mask * 255).astype(np.uint8)

    return mask_uint8


def make_prediction_filename(image_path, prompt):
    """Create filename: {image_id}__{prompt_with_underscores}.png"""
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    prompt_clean = prompt.replace(" ", "_").replace("/", "-")
    return f"{image_id}__{prompt_clean}.png"


def main():
    parser = argparse.ArgumentParser(description="Run CLIPSeg inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(BASE_DIR, "checkpoints", "best_model"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Binarization threshold"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to predict on"
    )
    args = parser.parse_args()

    # Load model
    processor, model = load_model(args.checkpoint)

    # Create output directory
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print("Running predictions")
    print(f"{'='*60}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Output: {PREDICTIONS_DIR}")

    total_predictions = 0

    for dataset_name, prompts in DATASET_PROMPTS.items():
        images = get_test_images(dataset_name, args.split)
        if not images:
            print(f"\n  [SKIP] No test images for {dataset_name}")
            continue

        print(f"\n  Dataset: {dataset_name} ({len(images)} images)")

        for prompt in prompts:
            print(f"    Prompt: '{prompt}'")
            dataset_pred_dir = os.path.join(PREDICTIONS_DIR, dataset_name)
            os.makedirs(dataset_pred_dir, exist_ok=True)

            for img_path in tqdm(images, desc=f"      Predicting", leave=False):
                image = Image.open(img_path).convert("RGB")
                mask = predict_single(
                    model, processor, image, prompt, args.threshold
                )

                # Save prediction
                filename = make_prediction_filename(img_path, prompt)
                save_path = os.path.join(dataset_pred_dir, filename)
                Image.fromarray(mask, mode="L").save(save_path)
                total_predictions += 1

    print(f"\n{'='*60}")
    print(f"Done! Generated {total_predictions} prediction masks")
    print(f"  Saved to: {PREDICTIONS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
