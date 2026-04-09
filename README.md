# Prompted Segmentation for Drywall QA

Training and inference pipeline for a text-conditioned segmentation model. Given an image and a natural-language prompt, it produces a binary mask for cracks or taping areas.

## Approach & Model

We fine-tuned the `CIDAS/clipseg-rd64-refined` decoder to output binary masks conditioned on text prompts. The CLIP encoder (vision and text) is kept frozen during training to retain its baseline zero-shot spatial generalization. We combined Binary Cross Entropy (BCE) and Dice Loss to improve the precision of the masks.

### Data
The pipeline uses two datasets from Roboflow:
- **Drywall-Join-Detect**: 820 train / 202 valid images. Used for "segment taping area", "segment joint/tape", "segment drywall seam". Ground truth bounding boxes were converted to segmentation masks.
- **Cracks-3ii36**: 5164 train / 201 valid / 4 test images. Used for "segment crack", "segment wall crack". Ground truth structural polygons were converted to precise segmentation masks.

## Setup & Execution

Install dependencies:
```bash
pip install -r requirements.txt
export ROBOFLOW_API_KEY="your_api_key_here"
```

1. **Download Data**: `python download_data.py`
2. **Train**: `python train.py --epochs 30 --batch_size 4 --lr 1e-4 --seed 42`
3. **Inference**: `python predict.py --threshold 0.5`
4. **Evaluate**: `python evaluate.py`

## Prediction Masks Format
Outputs are generated in `predictions/<dataset>/`.
- PNG, single-channel, same spatial size as source image, values {0, 255}.
- Filenames include image id and prompt, e.g. `image_id__segment_crack.png`.

## Results & Performance

Evaluations were performed on the validation set.

| Dataset  | Prompt                 | mIoU   | mDice  | N   |
|----------|------------------------|--------|--------|-----|
| cracks   | segment crack          | 0.4688 | 0.6152 | 201 |
| cracks   | segment wall crack     | 0.4660 | 0.6125 | 201 |
| drywall  | segment taping area    | 0.0884 | 0.1430 | 202 |
| drywall  | segment joint/tape     | 0.1411 | 0.2319 | 202 |
| drywall  | segment drywall seam   | 0.1891 | 0.3007 | 202 |

*Visual examples generated across datasets are saved in `results/visualizations/`.*

### Example Visualizations (Orig | GT | Pred)
Here is how the model performed on the validation set during a rapid, 2-epoch run:

**Crack Detection**
![Crack Example 1](results/visualizations/cracks/vis_1045-dat_png_jpg.rf.57f60467a0df171f05570845b0355a0c.png)
![Crack Example 2](results/visualizations/cracks/vis_1120-dat_png_jpg.rf.415d76182f1ec38f7949c18c273ef0e3.png)

**Drywall Seam Detection**
![Drywall Example 1](results/visualizations/drywall/vis_2000x1500_0_resized_jpg.rf.5ea90002e5c5e1cc2c092efe44e966cf.png)
![Drywall Example 2](results/visualizations/drywall/vis_2000x1500_15_resized_jpg.rf.22fc3512deed0691f3c2698aa76e7f0b.png)


### Footprint & Runtime
- **Model Size**: Base model contains ~150M parameters, with only 1.12M trainable parameters (frozen encoder).
- **Training Time**: ~2 minutes per epoch on an Apple Silicon MPS GPU backend using a time-constrained subset for rapid testing.
- **Inference Time**: ~3-5 milliseconds per image via MPS GPU backend.

### Failure Notes
- The drywall dataset originally only possessed bounding boxes instead of point-level polygons. This forces the model to segment an overly broad rectangular area for what is usually just a thin seam line, natively lowering mIoU/Dice scaling relative to the cracks dataset.
- Misclassifying long straight walls or shadows as cracks occasionally occurred due to the limited simulated training pass used in this test run.

*Note: All data generators and random operations explicitly utilize seed `42` for reproducibility across runs.*
