

Code and results for the paper **"How Stable Are Rice Leaf Disease Classifiers? A Cross-Validated Benchmark of Five Pretrained Models on the BRRI Dataset"**
(Jaimul Haque, Ezabul Alam; Department of CSE, BUBT, Dhaka, Bangladesh).

Contact: jaimul4067@gmail.com

## Overview

- Five ImageNet-pretrained models are fine-tuned under one identical pipeline: ResNet50, EfficientNet-B0, MobileNetV3-Large, ConvNeXt-Tiny, Swin-Tiny.
- Dataset: BRRI Rice Leaf Disease and Pest dataset (Hasan et al., Data in Brief 62, 111977, 2025). Only the 2,753 original images (7 classes) are used; augmented images are excluded.
- Evaluation: stratified 70/15/15 split (seed 42) with bootstrap CIs, and stratified 5-fold cross-validation with paired t-tests and Holm correction.
- Extras: majority-vote ensemble and ensemble size, focal-loss ablation, quantified Grad-CAM, inference benchmark, and inference-only external tests on the Sethy and RiceLeafBD datasets.
- Results are preliminary evidence (one seed per fold, five folds).

## Repository structure

```
notebooks/   Jupyter notebooks (run in order)
src/         dataset.py (sample gathering, stratified split, transforms, class weights)
results/     per-run outputs; results/cv/ holds per-fold JSON (test_idx, gts, preds)
figures/     figures used in the paper
data/        dataset folder (not included, see below)
```

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_EDA` | Dataset exploration |
| `02_Training`, `03_Training_EfficientNet`, `04_Training_MobileNetV3`, `05_Training_ConvNeXt`, `06_Training_SwinTiny` | Single-split training with class-weighted cross-entropy |
| `*_Focal` versions of 02-06 | Focal-loss ablation (gamma = 2) |
| `07_GradCAM` | Grad-CAM maps and background-mass analysis |
| `08_CrossValidation_ConvNeXt`, `08b_CrossValidation_Multi` | 5-fold cross-validation |
| `09_External_Test` | Inference-only test on external datasets |
| `10_Paper_Figures_Final` | Paper figures and the model size / speed benchmark |
| `11_Stats_Ensemble` | Paired tests and ensemble analysis |

## Data

1. Download the BRRI dataset (Hasan et al., Data in Brief 62, 111977, 2025).
2. Place the original (non-augmented) images under `data/Rice Dataset/Original Dataset/`, one folder per class.
3. The extra folder `Rice` (16 images) in the local copy is excluded from all experiments.

External test sets (not included): Sethy (Mendeley Data, doi 10.17632/fwcj7stb8r.1) and RiceLeafBD (Mendeley Data, doi 10.17632/kx9rx8p2mz.1).

## Setup

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib opencv-python pillow scipy jupyter
```

Python and library versions used for the paper: add here.

## Training settings

- Input 224x224; augmentation: horizontal flip, rotation 15 degrees, ColorJitter 0.2; ImageNet normalisation.
- Adam, lr 1e-4, no weight decay, batch size 32, up to 30 epochs, ReduceLROnPlateau (factor 0.5, patience 3), early stopping (patience 7), best validation-loss checkpoint.
- Loss: class-weighted cross-entropy, w_c = N / (K n_c); focal loss uses gamma = 2 with the same weights.
- Weights: ConvNeXt, EfficientNet, Swin use IMAGENET1K_V1; ResNet50, MobileNetV3 use IMAGENET1K_V2; all layers fine-tuned.
- FP32 everywhere, except the single-split focal ConvNeXt-Tiny run (batch size 8, mixed precision).
- GPU: NVIDIA RTX 3050.

## Main results (test, Macro-F1)

| Model | Single split | 5-fold CV (mean +- SD) |
|---|---|---|
| ConvNeXt-Tiny | 0.7215 | 0.7801 +- 0.0175 |
| ResNet50 | 0.7346 | 0.7435 +- 0.0254 |
| EfficientNet-B0 | 0.7483 | 0.7429 +- 0.0136 |
| Swin-Tiny | 0.7682 | 0.7417 +- 0.0288 |
| MobileNetV3-Large | 0.7234 | 0.7303 +- 0.0271 |
| Ensemble (5 models) | 0.7866 (softmax avg.) | 0.797 +- 0.019 (majority vote) |

ConvNeXt-Tiny ranks last on the single split but first under cross-validation. See the paper for confidence intervals, statistical tests, per-class results, external validation and limitations.

## Citation

If you use this code, please cite the paper (reference to be added after publication).

## License

Add a license file (for example MIT) and state it here.
