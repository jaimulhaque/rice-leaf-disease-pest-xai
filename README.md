# Rice Leaf Disease and Pest Classification (XAI)

Explainable deep learning for multi-class rice leaf disease and pest
classification using transfer learning, developed for [Conference Name] 2026.

## Overview
This project fine-tunes five CNN/Transformer architectures to classify
rice leaf images into 7 categories (Healthy, Rice Blast, Scald,
Leaf-folder Injury, Insect Infestation, Rice Stripes, Tungro), and uses
Grad-CAM to explain model predictions.

## Dataset
- Source: Rice Leaf Disease and Pest Dataset (BRRI, Gazipur), Mendeley
- Link: https://data.mendeley.com/datasets/vwv3nry3wr/1
- Used: 2,753 original images only (augmented images excluded to avoid
  data leakage)
- 7 classes, resolution up to 4064x3048

## Models
- ResNet50
- EfficientNet-B0
- MobileNetV3-Large
- ConvNeXt-Tiny
- Swin-Tiny

## Setup
\`\`\`bash
git clone https://github.com/<username>/rice-leaf-disease-pest-xai.git
cd rice-leaf-disease-pest-xai
pip install -r requirements.txt
\`\`\`

Place the downloaded `original/` folder into `data/raw/`.

## Pipeline
1. `notebooks/01_eda.ipynb` — exploratory data analysis
2. `notebooks/02_preprocessing.ipynb` — split, clean, augmentation
3. `notebooks/03-07_train_*.ipynb` — model training
4. `notebooks/08_evaluation.ipynb` — metrics, confusion matrix
5. `notebooks/09_gradcam_xai.ipynb` — explainability

## Results
| Model | Accuracy | Macro-F1 | Params | Inference (ms) |
|---|---|---|---|---|
| ResNet50 | - | - | - | - |
| EfficientNet-B0 | - | - | - | - |
| MobileNetV3-Large | - | - | - | - |
| ConvNeXt-Tiny | - | - | - | - |
| Swin-Tiny | - | - | - | - |

## Citation
If you use this code, please cite our paper (details to be added upon
publication).

## License
MIT
